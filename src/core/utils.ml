open Result
module IntMap = Map.Make (Int)
module StringMap = Map.Make (String)
module IntSet = Set.Make (Int)
module StringSet = Set.Make (String)

let ( let* ) = bind
let id x = x
let keys t = IntMap.bindings t |> List.split |> fun (indices, _) -> IntSet.of_list indices
