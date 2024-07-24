
local function map(arr, fn)
    local new = {}
    for i = 1, #arr, 1 do
        new[i] = fn(arr[i])
    end
    return new
end
--@reveal map

local arr1 = {1, 2, 3}
--@reveal arr1
local arr2 = map(arr1, function(x)
    return tostring(x)
end)
--@reveal arr2
local arr3 = map(arr2, function(x)
    return tonumber(x)
end)
--@reveal arr3

local x = arr3[1]
if x == 123 then
  --@reveal x
end