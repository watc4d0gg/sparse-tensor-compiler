open Result

let ( let* ) = bind

module Map = Map.Make (String)
module Set = Set.Make (String)

exception SyntaxError of string

type op =
  | ADD
  | SUB
  | MUL

type index =
  | Var of string
  | Literal of int
  | Term of int * string
  | Affine of index * index

type expr =
  | Constant of float
  | Access of string * index list
  | Unary of op * expr
  | Binary of op * expr * expr

(* TODO: add format type *)
type stmt =
  | Format of string * string
  | Assign of string * index list * expr

type ast = stmt list

let fmt_op = function
  | ADD -> "+"
  | SUB -> "-"
  | MUL -> "*"

let rec fmt_index = function
  | Var i -> i
  | Literal l -> Int.to_string l
  | Term (c, i) -> Printf.sprintf "%d * %s" c i
  | Affine (i1, i2) -> Printf.sprintf "%s + %s" (fmt_index i1) (fmt_index i2)

let rec fmt_expr = function
  | Constant c -> Printf.sprintf "%f" c
  | Access (id, indices) ->
      Printf.sprintf "%s%s"
        id
        (if List.is_empty indices then "" else ("(" ^ String.concat ", " (List.map fmt_index indices) ^ ")"))
  | Unary (op, e) -> Printf.sprintf "%s%s" (fmt_op op) (fmt_expr e)
  | Binary (op, e1, e2) -> Printf.sprintf "%s %s %s" (fmt_expr e1) (fmt_op op) (fmt_expr e2)

let fmt_stmt = function
  | Format (id, format) -> Printf.sprintf "%s : %s" id format
  | Assign (id, indices, e) ->
    Printf.sprintf "%s%s = %s"
      id
      (if List.is_empty indices then "" else ("(" ^ String.concat ", " (List.map fmt_index indices) ^ ")"))
      (fmt_expr e)

let rec check_tensors existing_tensors = function
  | [] -> ok ()
  | stmt :: stmts ->
    (match stmt with
     | Format (id, _) -> check_tensors (Set.add id existing_tensors) stmts
     | Assign (id, _, e2) ->
       let updated = Set.add id existing_tensors in
       let rec exists = function
         | Access (id, _) ->
           if Set.mem id updated
           then ok ()
           else error (Printf.sprintf "Undefined tensor %s" id)
         | Unary (_, e) -> exists e
         | Binary (_, e1, e2) ->
           let* _ = exists e1 in
           exists e2
         | _ -> ok ()
       in
       let* _ = exists e2 in
       check_tensors updated stmts)
;;

let rec check_formats initialized_formats = function
  | [] -> ok ()
  | stmt :: stmts ->
    (match stmt with
     | Format (id, fmt) ->
       (match Map.find_opt id initialized_formats with
        | Some current ->
          if String.equal fmt current
          then ok ()
          else
            error
              (Printf.sprintf "Format for %s reinitialized from %s to %s" id current fmt)
        | None -> check_formats (Map.add id fmt initialized_formats) stmts)
     | _ -> check_formats initialized_formats stmts)
;;

(* TODO: infer affine maps
  - If an index variable is defined in the iteration space
    but does not appear in the expression of a tensor operand, the tensor operand
    is broadcast along the dimension represented by the index variable.
  - If an index variable appears
    in the expression of some tensor operands but not in the expression of the destination tensor,
    then the corresponding dimension is reduced on the smallest sub-expression that captures the use
    of the index variable. *)
