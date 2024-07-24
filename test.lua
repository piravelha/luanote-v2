
local function Array(values)
  return {
    values = values,
    map = function(self, fn)
      local new = {}
      for i = 1, #(self.values), 1 do
        new[i] = fn(self.values[i])
      end
      return Array(new)
    end,
  }
end

local function rec(x)
  return rec
end

--@reveal Array({1}).map
