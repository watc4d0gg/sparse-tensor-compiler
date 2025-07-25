open Mlir
open Mlir.Ir
open Environment
open Utils

exception SyntaxError of string

type tensor_id = string
type subscript = AffineExpr.t list

let has_dense_semantics = function
  | LevelType.Dense _ | LevelType.Batch _ -> true
  | _ -> false

let rec sparse_admissible = function
  | AffineExpr.Dim (_, _) as expr -> expr
  | AffineExpr.Add (l, r, raw) as expr ->
    let l' =
      try sparse_admissible l with
      | SyntaxError _ ->
        SyntaxError
          (Printf.sprintf
             "\"%s\" is an unnsupported index expression for indexing a sparse level!"
             (print_as_string AffineExpr.print expr))
        |> raise
    in
    let r' =
      try sparse_admissible r with
      | SyntaxError _ ->
        SyntaxError
          (Printf.sprintf
             "\"%s\" is an unnsupported index expression for indexing a sparse level!"
             (print_as_string AffineExpr.print expr))
        |> raise
    in
    AffineExpr.Add (l', r', raw)
  | AffineExpr.Mul (l, r, _) as expr ->
    (match l, r with
     | AffineExpr.Constant (_, _), AffineExpr.Dim (_, _)
     | AffineExpr.Dim (_, _), AffineExpr.Constant (_, _) -> expr
     | _ ->
       SyntaxError
         (Printf.sprintf
            "\"%s\" is an unnsupported index expression for indexing a sparse level!"
            (print_as_string AffineExpr.print expr))
       |> raise)
  | expr ->
    SyntaxError
      (Printf.sprintf
         "\"%s\" is an unnsupported index expression for indexing a sparse level!"
         (print_as_string AffineExpr.print expr))
    |> raise

module TensorExpr = struct
  type op =
    | ADD
    | SUB
    | MUL

  type t =
    [ `Constant of float
    | `Access of tensor_id * subscript
    | `Unary of op * t
    | `Binary of op * t * t
    ]

  let rec debug_repr =
    let repr_op = function
      | ADD -> "+"
      | SUB -> "-"
      | MUL -> "*"
    in
    function
    | `Constant c -> Printf.sprintf "(Const %f)" c
    | `Access (tensor, subscript) ->
      if List.is_empty subscript
      then "(Access " ^ tensor ^ ")"
      else
        Printf.sprintf
          "(Access %s(%s))"
          tensor
          (List.map (print_as_string AffineExpr.print) subscript |> String.concat ", ")
    | `Unary (op, e) -> Printf.sprintf "(Unary (%s) %s)" (repr_op op) (debug_repr e)
    | `Binary (op, e1, e2) ->
      Printf.sprintf "(Binary (%s) %s %s)" (debug_repr e1) (repr_op op) (debug_repr e2)

  let rec repr =
    let repr_op = function
      | ADD -> "+"
      | SUB -> "-"
      | MUL -> "*"
    in
    function
    | `Constant c -> Printf.sprintf "%f" c
    | `Access (tensor, subscript) ->
      if List.is_empty subscript
      then tensor
      else
        Printf.sprintf
          "%s(%s)"
          tensor
          (List.map (print_as_string AffineExpr.print) subscript |> String.concat ", ")
    | `Unary (op, e) -> Printf.sprintf "%s%s" (repr_op op) (repr e)
    | `Binary (op, e1, e2) -> Printf.sprintf "%s %s %s" (repr e1) (repr_op op) (repr e2)
end

module Einsum (Env : Environment) = struct
  include Env

  type t =
    [ `Declare of tensor_id * SparseTensorEncodingAttr.t
    | `Assign of [ `Access of tensor_id * subscript ] * TensorExpr.t
    | `Sequence of t * t
    ]

  let named_formats =
    StringMap.of_list
      [ ( "CSR"
        , SparseTensorEncodingAttr.get
            context
            [ LevelType.dense (); LevelType.compressed [] ]
            (AffineMap.get
               context
               2
               0
               [ AffineExpr.dim context 0; AffineExpr.dim context 1 ])
            None
            None
            None
            None
            None )
      ; ( "CSC"
        , SparseTensorEncodingAttr.get
            context
            [ LevelType.dense (); LevelType.compressed [] ]
            (AffineMap.get
               context
               2
               0
               [ AffineExpr.dim context 1; AffineExpr.dim context 0 ])
            None
            None
            None
            None
            None )
      ]

  let rec repr =
    let repr_prop = function
      | LevelNonDefaultProperty.NonOrdered -> "!O"
      | LevelNonDefaultProperty.NonUnique -> "!U"
      | LevelNonDefaultProperty.StructureOfArrays -> "SoA"
    in
    let repr_props props =
      if LevelNonDefaultProperties.cardinal props == 0
      then ""
      else
        Printf.sprintf
          "(%s)"
          (LevelNonDefaultProperties.elements props
           |> List.map repr_prop
           |> String.concat ", ")
    in
    let repr_level_type = function
      | LevelType.Dense _ -> "Dense"
      | LevelType.Batch _ -> "Batch"
      | LevelType.Compressed (props, _) ->
        Printf.sprintf "Compressed%s" (repr_props props)
      | LevelType.LooseCompressed (props, _) ->
        Printf.sprintf "LooseCompressed%s" (repr_props props)
      | LevelType.Singleton (props, _) -> Printf.sprintf "Singleton%s" (repr_props props)
      | LevelType.Structured (n, m, _) -> Printf.sprintf "Structured[%d:%d]" n m
    in
    let repr_format format =
      match
        StringMap.find_first_opt
          (fun name -> Attribute.equal format (StringMap.find name named_formats))
          named_formats
      with
      | Some (name, _) -> "#" ^ name
      | None ->
        let dim_to_lvl = format#dimensions_to_levels in
        Printf.sprintf
          "Tensor{(%s) -> Levels{%s}}"
          (List.init dim_to_lvl#dims (Printf.sprintf "d%d") |> String.concat ", ")
          (List.init dim_to_lvl#results (fun i ->
             Printf.sprintf
               "(%s) : %s"
               (dim_to_lvl#result i |> print_as_string AffineExpr.print)
               (format#level_type i |> repr_level_type))
           |> String.concat ", ")
    in
    function
    | `Declare (tensor, format) -> Printf.sprintf "%s : %s" tensor (repr_format format)
    | `Assign (`Access (tensor, subscript), expr) ->
      Printf.sprintf
        "%s%s = %s"
        tensor
        (if List.is_empty subscript
         then ""
         else
           "("
           ^ (List.map (print_as_string AffineExpr.print) subscript |> String.concat ", ")
           ^ ")")
        (TensorExpr.repr expr)
    | `Sequence (e1, e2) -> Printf.sprintf "%s\n%s" (repr e1) (repr e2)

  (** [tensors ast] returns a map of all tensors and their formats declared in the program *)
  let rec tensors = function
    | `Declare (tensor, format) -> StringMap.singleton tensor format
    | `Sequence (first, second) ->
      StringMap.union
        (fun tensor _ _ ->
           SyntaxError (Printf.sprintf "Tensor %s is initialized twice!" tensor) |> raise)
        (tensors first)
        (tensors second)
    | _ -> StringMap.empty

  (** [assignments ast] returns all the tensor assignments (kernels) in the given program *)
  let rec assignments = function
    | `Assign (access, expr) -> [ `Assign (access, expr) ]
    | `Sequence (first, second) -> assignments first @ assignments second
    | _ -> []

  (** [accesses assignment] returns all the tensor accesses in the given tensor assignment *)
  let accesses (`Assign (access, expr)) =
    let rec collect_accesses = function
      | `Access (tensor, subscript) -> [ `Access (tensor, subscript) ]
      | `Unary (_, expr) -> collect_accesses expr
      | `Binary (_, e1, e2) -> collect_accesses e1 @ collect_accesses e2
      | _ -> []
    in
    access :: collect_accesses expr

  (** [num_loops assignment] returns the number of loops (index variables) contained in the tensor assignment *)
  let num_loops (`Assign ((`Access (_, _) as access), expr)) =
    let rec count_loops = function
      | `Access (_, subscript) ->
        List.fold_left
          (fun curr next ->
             let rec max_index = function
               | AffineExpr.Dim (i, _) -> max curr i
               | AffineExpr.Add (l, r, _) -> max (max_index l) (max_index r)
               | AffineExpr.Mul (l, r, _) -> max (max_index l) (max_index r)
               | _ -> curr
             in
             max_index next)
          0
          subscript
      | `Unary (_, expr) -> count_loops expr
      | `Binary (_, e1, e2) -> max (count_loops e1) (count_loops e2)
      | _ -> 0
    in
    max (count_loops access) (count_loops expr) + 1

  let parallel_loops (`Assign (`Access (_, subscript), _)) =
    List.fold_left
      (fun set -> function
         | AffineExpr.Dim (i, _) -> IntSet.add i set
         | _ -> set)
      IntSet.empty
      subscript

  (** [dependent_indices access] returns a map of dependent indices and their coefficients at each level addressed by the given access *)
  let dependent_indices (`Access (_, subscript)) =
    let rec collect_indices = function
      | AffineExpr.Dim (i, _) -> IntMap.singleton i 1
      | AffineExpr.Add (l, r, _) ->
        IntMap.union
          (fun _ fst snd -> Some (fst + snd))
          (collect_indices l)
          (collect_indices r)
      | AffineExpr.Mul (l, r, _) ->
        (match l, r with
         | AffineExpr.Constant (c, _), AffineExpr.Dim (i, _)
         | AffineExpr.Dim (i, _), AffineExpr.Constant (c, _) -> IntMap.singleton i c
         | _ -> IntMap.empty)
      | _ -> IntMap.empty
    in
    List.map collect_indices subscript

  let indexing_maps assignment =
    let num_loops = num_loops assignment in
    List.map
      (fun (`Access (_, subscript)) -> AffineMap.get context num_loops 0 subscript)
      (accesses assignment)

  (** [validate prog] validates all tensor declarations and all tensor accesses in the program
      and returns the program itself if successful, otherwise raises an SyntaxError
      when a tensor is initialized twice or an access points to an undefined tensor *)
  let validate prog =
    let tensors = tensors prog in
    let validate_access is_output (`Access (tensor, subscript)) =
      if StringMap.mem tensor tensors |> not
      then SyntaxError (Printf.sprintf "Undefined tensor %s!" tensor) |> raise
      else (
        let format = StringMap.find tensor tensors in
        let access_rank = List.length subscript in
        if format#level_rank != access_rank
        then
          SyntaxError
            (Printf.sprintf
               "Mismatched access to a %d-dimensional tensor %s, used %d indices"
               format#level_rank
               tensor
               access_rank)
          |> raise
        else if
          is_output
          && List.for_all
               (function
                 | AffineExpr.Dim (_, _) -> true
                 | _ -> false)
               subscript
             |> not
        then
          SyntaxError
            (Printf.sprintf
               "The assignment to the output tensor %s only accepts a non-affine access, \
                i.e. pure index variable subscript!"
               tensor)
          |> raise
        else
          `Access
            ( tensor
            , List.mapi
                (fun i index ->
                   if format#level_type i |> has_dense_semantics
                   then index
                   else sparse_admissible index)
                subscript ))
    in
    let rec validate_expr = function
      | `Access (tensor, subscript) -> validate_access false (`Access (tensor, subscript))
      | `Unary (op, e) -> `Unary (op, validate_expr e)
      | `Binary (op, e1, e2) -> `Binary (op, validate_expr e1, validate_expr e2)
      | expr -> expr
    in
    let rec validate_ast = function
      | `Assign (access, expr) -> `Assign (validate_access true access, validate_expr expr)
      | `Sequence (first, second) ->
        let first' = validate_ast first
        and second' = validate_ast second in
        `Sequence (first', second')
      | stmt -> stmt
    in
    validate_ast prog
end

(* let rec collect_ *)
(* TODO: infer affine maps
  - If an index variable is defined in the iteration space (of the destination tensor)
    but does not appear in the expression of a tensor operand, the tensor operand
    is broadcast along the dimension represented by the index variable.
  - If an index variable appears in the expression of some tensor operands
    but not in the expression of the destination tensor,
    then the corresponding dimension is reduced on the smallest sub-expression that captures the use
    of the index variable, i.e.
      Not D(i) = sum(j, A(i, j) + B(i, j), broadcast(j, C(i))), but D(i) = sum(j, A(i, j) + B(i, j)) + C(i)*)
