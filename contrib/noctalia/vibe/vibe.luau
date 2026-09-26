--!nonstrict
-- /vibe launcher provider: semantic search over the local library via `vibe --json`.

local lastResults = {}

local function tr(key, args)
	return noctalia.tr(key, args)
end

local function shellQuote(s)
	return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function vibeExe()
	local configured = noctalia.getConfig("vibe_path")
	if type(configured) == "string" and configured ~= "" then
		return noctalia.expandPath(configured)
	end
	if noctalia.commandExists("vibe") then
		return "vibe"
	end
	local fallback = noctalia.expandPath("~/.local/bin/vibe")
	if noctalia.commandExists(fallback) then
		return fallback
	end
	return nil
end

local function maxResults()
	local n = noctalia.getConfig("max_results")
	return (type(n) == "number" and n > 0) and math.floor(n) or 15
end

local function player()
	local p = noctalia.getConfig("player")
	return (type(p) == "string" and p ~= "") and p or "mpv"
end

local function playArgs(exe)
	return shellQuote(exe) .. " --detach --player " .. shellQuote(player())
end

function onQuery(query)
	local text = noctalia.string.trim(query)
	if text == "" then
		launcher.setResults(query, {
			{ id = "", title = tr("hint"), subtitle = tr("hint-subtitle"), glyph = "music" },
		})
		return
	end

	local exe = vibeExe()
	if exe == nil then
		launcher.setResults(query, {
			{ id = "", title = tr("not-installed"), subtitle = tr("not-installed-subtitle"), glyph = "alert-triangle" },
		})
		return
	end

	launcher.setResults(query, {
		{ id = "", title = tr("searching"), subtitle = tr("searching-subtitle", { query = text }), glyph = "loader" },
	})

	local cmd = string.format("%s --json -n %d %s", shellQuote(exe), maxResults(), shellQuote(text))
	noctalia.runAsync(cmd, function(result)
		if result.exitCode ~= 0 then
			local msg = noctalia.string.trim(result.stderr or "")
			msg = msg:gsub("^error: ", "")
			launcher.setResults(query, {
				{ id = "", title = tr("error"), subtitle = msg ~= "" and msg or ("exit " .. tostring(result.exitCode)), glyph = "alert-triangle" },
			})
			return
		end

		local ok, hits = pcall(noctalia.json.decode, result.stdout or "")
		if not ok or type(hits) ~= "table" or #hits == 0 then
			launcher.setResults(query, {
				{ id = "", title = tr("no-results"), subtitle = tr("no-results-subtitle"), glyph = "music-off" },
			})
			return
		end

		lastResults = {}
		local rows = {
			{
				id = "all:" .. text,
				title = tr("play-all", { count = #hits }),
				subtitle = tr("play-all-subtitle", { query = text, player = player() }),
				glyph = "player-play",
			},
		}
		for i, hit in ipairs(hits) do
			local id = "track:" .. tostring(i)
			lastResults[id] = hit.path
			table.insert(rows, {
				id = id,
				title = hit.name,
				subtitle = string.format("%s  ·  %.2f", hit.dir, hit.score),
				glyph = "music",
			})
		end
		launcher.setResults(query, rows)
	end)
end

function onActivate(id)
	if id == nil or id == "" then
		return
	end
	local exe = vibeExe()
	if exe == nil then
		return
	end

	local text = id:match("^all:(.+)$")
	if text ~= nil then
		local cmd = string.format("%s -n %d %s", playArgs(exe), maxResults(), shellQuote(text))
		noctalia.runAsync(cmd .. " </dev/null >/dev/null 2>&1")
		return
	end

	local path = lastResults[id]
	if path ~= nil then
		noctalia.runAsync(playArgs(exe) .. " --play-path " .. shellQuote(path) .. " </dev/null >/dev/null 2>&1")
	end
end
