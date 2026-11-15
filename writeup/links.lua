-- Relative repository links point to GitHub, so they work in the PDF and HTML.
local base = "https://github.com/ValueShift/lexi_hack/blob/main/"
function Link(el)
  local t = el.target
  if not t:match("^%a[%w+.-]*:") and not t:match("^#") then
    el.target = base .. t:gsub("^%./", "")
  end
  return el
end

-- GitHub display math is written as ```math fences; typeset them as display equations.
function CodeBlock(el)
  if el.classes:includes("math") then
    return pandoc.Para({ pandoc.Math("DisplayMath", el.text) })
  end
end

-- The "Also as: PDF · HTML" line only makes sense on GitHub; drop it from the PDF and HTML.
function Para(el)
  local first = el.content[1]
  if first and first.t == "Strong" and pandoc.utils.stringify(first):match("^Also as") then
    return {}
  end
end

-- Style the two call-out boxes (content warning, "For judges") as distinct panels.
function BlockQuote(el)
  local text = pandoc.utils.stringify(el)
  if text:match("^Content warning") then
    return pandoc.Div(el.content, pandoc.Attr("", {"callout", "warning"}))
  elseif text:match("^For judges") then
    return pandoc.Div(el.content, pandoc.Attr("", {"callout", "judges"}))
  end
end
