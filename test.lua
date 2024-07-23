
local function map(arr, fn)
    local new = {}
    for i = 1, #arr, 1 do
        new[i] = fn(arr[i])
    end
    return new
end

local arr1 = {1, 2, 3}
local arr2 = map(arr1, function(x)
    return tostring(x)
end)
local arr3 = map(arr2, function(x)
    return tonumber(x)
end)

--@reveal arr2
