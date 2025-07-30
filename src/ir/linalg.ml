open Mlir
open Mlir.Ir
open Core
open! Tensor_runtime

exception LoweringError of string

module Linalg (Env : Environment) = struct
  include Env
  module Einsum = Einsum (Env)

  module IterationGraph = struct
    type node =
      { in_degree : int
      ; out : IntSet.t
      }

    type t = node IntMap.t * node IntMap.t
    (* the sparse subgraph and the complete graph, both consisting of nodes
       which store their degree of incoming edges and their adjacent vertices *)

    let from tensors assignment =
      let rec add_loops g = function
        | 0 -> g
        | n ->
          (match IntMap.find_opt (n - 1) g with
           | Some _ -> add_loops g (n - 1)
           | None ->
             IntMap.add
               (n - 1)
               { in_degree = 0; out = IntSet.empty }
               (add_loops g (n - 1)))
      in
      let rec add_constraints g = function
        | [] -> g
        | (from, to_) :: constraints ->
          add_constraints
            (IntSet.fold
               (fun index g ->
                  let { in_degree; out } = IntMap.find index g in
                  let unconnected = IntSet.diff to_ out in
                  IntSet.fold
                    (fun index g ->
                       match IntMap.find_opt index g with
                       | Some { in_degree; out } ->
                         IntMap.add index { in_degree = in_degree + 1; out } g
                       | None -> IntMap.add index { in_degree = 1; out = IntSet.empty } g)
                    unconnected
                    (IntMap.add index { in_degree; out = IntSet.union out to_ } g))
               from
               g)
            constraints
      in
      let num_loops = Einsum.num_loops assignment in
      let accesses = Einsum.accesses assignment in
      List.fold_left
        (fun (sparse, total) (`Access (tensor, _) as access) ->
           let format = StringMap.find tensor tensors in
           let level_types = List.init format#level_rank format#level_type in
           let dependent_indices = Einsum.dependent_indices access in
           let _, constraints =
             List.fold_left
               (fun (from, acc) next ->
                  let to_ = keys next in
                  to_, (from, to_) :: acc)
               (List.hd dependent_indices |> keys, [])
               (List.tl dependent_indices)
           in
           (* List.iter (fun (from, to_) -> Printf.printf "%s -> %s" (IntSet.to_list from |> List.map Int.to_string |> String.concat " | ") (IntSet.to_list to_ |> List.map Int.to_string |> String.concat " | ")) constraints; *)
           if List.for_all has_dense_semantics level_types
           then sparse, add_constraints total constraints
           else add_constraints sparse constraints, add_constraints total constraints)
        (add_loops IntMap.empty num_loops, add_loops IntMap.empty num_loops)
        (List.tl accesses)

    let loop_order (sparse, total) =
      let rec topo_sort g =
        match
          IntMap.find_first_opt (fun index -> (IntMap.find index g).in_degree == 0) g
        with
        | Some (index, { out; _ }) ->
          index
          :: topo_sort
               (IntSet.fold
                  (fun index g ->
                     IntMap.add
                       index
                       (let { in_degree; out } = IntMap.find index g in
                        { in_degree = in_degree - 1; out })
                       g)
                  out
                  (IntMap.remove index g))
        | None -> if IntMap.is_empty g then [] else SyntaxError "Cycle detected!" |> raise
      in
      let sparse_order = topo_sort sparse
      and total_order = topo_sort total in
      sparse_order
      @ List.filter (fun index -> List.mem index sparse_order |> not) total_order
  end

  let rec to_arith location body output_rank access_counter = function
    | `Constant c ->
      let const =
        Arith.constant (FloatAttr.get element_type c location) location context
      in
      body#append_operation const;
      (const#result 0 :> Value.t), 0
    | `Access (_, subscript) ->
      let index = !access_counter in
      access_counter := index + 1;
      (body#argument index :> Value.t), List.length subscript
    | `Unary (op, expr) as unary ->
      let value, rank = to_arith location body output_rank access_counter expr in
      (match op with
       | TensorExpr.ADD -> value, rank
       | TensorExpr.SUB ->
         let negation = Arith.negf (body#argument 0) location in
         body#append_operation negation;
         (negation#result 0 :> Value.t), rank
       | _ ->
         LoweringError
           (Printf.sprintf
              "\"%s\"is not a valid tensor expression!"
              (TensorExpr.repr unary))
         |> raise)
    | `Binary (op, left, right) ->
      let left_value, left_rank = to_arith location body output_rank access_counter left
      and right_value, right_rank =
        to_arith location body output_rank access_counter right
      in
      let left_value =
        if left_rank > output_rank && left_rank > right_rank
        then (
          let zero =
            Arith.constant (FloatAttr.get element_type 0.0 location) location context
          in
          body#append_operation zero;
          let reduction =
            SparseTensor.reduce
              left_value
              (zero#result 0)
              (zero#result 0)
              location
              ~init:(fun body -> [ body#argument 0 ])
          in
          body#append_operation reduction;
          (reduction#result 0 :> Value.t))
        else left_value
      and right_value =
        if right_rank > output_rank && left_rank < right_rank
        then (
          let zero =
            Arith.constant (FloatAttr.get element_type 0.0 location) location context
          in
          body#append_operation zero;
          let reduction =
            SparseTensor.reduce
              right_value
              (zero#result 0)
              (zero#result 0)
              location
              ~init:(fun body -> [ body#argument 0 ])
          in
          body#append_operation reduction;
          (reduction#result 0 :> Value.t))
        else right_value
      in
      let operation =
        match op with
        | TensorExpr.ADD -> Arith.addf left_value right_value location
        | TensorExpr.SUB -> Arith.subf left_value right_value location
        | TensorExpr.MUL -> Arith.mulf left_value right_value location
      in
      body#append_operation operation;
      ( (operation#result 0 :> Value.t)
      , if left_rank = 0
        then right_rank
        else if right_rank = 0
        then left_rank
        else min left_rank right_rank )

  let lower ast =
    let tensors = Einsum.tensors ast
    and kernels = Einsum.assignments ast in
    let input_order, output_order = Einsum.inputs_and_outputs kernels in
    with_module ~init:(fun body ->
      let dense_output_order =
        List.filter
          (fun output -> (StringMap.find output tensors)#encoding |> Option.is_none)
          output_order
      in
      body#append_operation
      @@ Func.func
           context
           (StringAttr.get context "kernels")
           (FunctionType.get
              context
              (List.map (fun input -> StringMap.find input tensors) input_order
               @ List.map (fun output -> StringMap.find output tensors) dense_output_order
              )
              (List.map (fun name -> StringMap.find name tensors) output_order)
            |> TypedAttr.get)
           []
           location
           ~init:(fun body ->
             let last_output_results =
               ref
                 (List.mapi
                    (fun i output ->
                       output, (body#argument (List.length input_order + i) :> Value.t))
                    dense_output_order
                  |> StringMap.of_list)
             in
             List.iter
               (fun (`Assign (`Access (output, _), expr) as kernel) ->
                  let num_loops = Einsum.num_loops kernel
                  and parallel_loops = Einsum.parallel_loops kernel
                  and accesses = Einsum.accesses kernel
                  and indexing_maps = Einsum.indexing_maps kernel in
                  let output_value =
                    match StringMap.find_opt output !last_output_results with
                    | Some result ->
                      (* use last result *)
                      result
                    | None ->
                      (* initialize empty output tensor *)
                      let output_type = StringMap.find output tensors in
                      (* DICLAIMER: this will get cleaned up by cse/canonicalize passes *)
                      let dims =
                        List.init output_type#rank (fun i ->
                          let constant =
                            Arith.constant (IntegerAttr.get index_type i) location context
                          in
                          body#append_operation constant;
                          constant#result 0)
                      in
                      let empty = Tensor.empty dims output_type location in
                      body#append_operation empty;
                      (* remember the last result for this output tensor *)
                      let last_result = (empty#result 0 :> Value.t) in
                      last_output_results
                      := StringMap.add output last_result !last_output_results;
                      last_result
                  in
                  let input_values =
                    List.tl accesses
                    |> List.map (fun (`Access (input, _)) ->
                      match List.find_index (String.equal input) input_order with
                      | Some index -> (body#argument index :> Value.t)
                      | None -> (StringMap.find input !last_output_results :> Value.t))
                  in
                  let kernel =
                    Linalg.generic
                      context
                      input_values
                      [ output_value ]
                      (List.tl indexing_maps @ [ List.hd indexing_maps ])
                      (List.init num_loops (fun i ->
                         if IntSet.mem i parallel_loops then "parallel" else "reduction"))
                      location
                      ~init:(fun body ->
                        let value, _ =
                          to_arith
                            location
                            body
                            (IntSet.cardinal parallel_loops)
                            (ref 0)
                            expr
                        in
                        let reduce =
                          Arith.addf
                            value
                            (body#argument (List.length input_values))
                            location
                        in
                        body#append_operation reduce;
                        [ (reduce#result 0 :> Value.t) ])
                  in
                  body#append_operation kernel;
                  last_output_results
                  := StringMap.add
                       output
                       (kernel#result 0 :> Value.t)
                       !last_output_results)
               kernels;
             List.map
               (fun output -> StringMap.find output !last_output_results)
               output_order))
end

(* let num_loops = Einsum.num_loops kernel
                    and parallel_loops = Einsum.parallel_loops kernel
                    and accesses = Einsum.accesses kernel
                    and indexing_maps = Einsum.indexing_maps kernel in
                    (* let output_value =
                      match StringMap.find_opt output !last_output_results with
                      | Some result ->
                        (* use last result *)
                        result
                      | None ->
                        (* initialize empty output tensor *)
                        let output_type = StringMap.find output tensors in
                        (* DICLAIMER: this will get cleaned up by cse/canonicalize passes *)
                        let dims =
                          List.init output_type#rank (fun i ->
                            let constant =
                              Arith.constant (IntegerAttr.get index i) location context
                            in
                            body#append_operation constant;
                            constant#result 0)
                        in
                        let empty = Tensor.empty dims output_type location in
                        body#append_operation empty;
                        (* remember the last result for this output tensor *)
                        let last_result = empty#result 0 in
                        last_output_results
                        := StringMap.add output last_result !last_output_results;
                        last_result
                    in
                    let input_values =
                      List.tl accesses
                      |> List.map (fun (`Access (input, _)) ->
                        match List.find_index (String.equal input) input_order with
                        | Some index -> (body#argument index :> Value.t)
                        | None ->
                          (StringMap.find input !last_output_results :> Value.t))
                    in
                    let kernel =
                      Linalg.generic
                        context
                        input_values
                        [ output_value ]
                        (List.tl indexing_maps @ [ List.hd indexing_maps ])
                        (List.init num_loops (fun i ->
                          if IntSet.mem i parallel_loops
                          then "parallel"
                          else "reduction"))
                        location
                        ~init:(fun body ->
                          let value, _ =
                            to_arith
                              location
                              body
                              (IntSet.cardinal parallel_loops)
                              (ref 0)
                              expr
                          in
                          let reduce =
                            Arith.addf
                              value
                              (body#argument (List.length input_values))
                              location
                          in
                          body#append_operation reduce;
                          [ (reduce#result 0 :> Value.t) ])
                    in
                    body#append_operation kernel; *)
                    [ (kernel#result 0 :> Value.t) ] *)
