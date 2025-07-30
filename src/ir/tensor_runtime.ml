open Mlir
open Mlir.Ir
open Core

module TensorRuntime (Env : Environment) = struct
  include Env
  module Einsum = Einsum (Env)

  let generate_init input_name input_type =
    Func.func
      context
      (StringAttr.get context ("init_" ^ input_name))
      (FunctionType.get context [] [ input_type ] |> TypedAttr.get)
      [ Identifier.get context "llvm.emit_c_interface", Attribute.unit context ]
      location
      ~init:(fun body ->
        let init = Tensor.empty [] input_type location in
        body#append_operation init;
        [ init#result 0 ])

  let generate_insert input_name input_type =
    Func.func
      context
      (StringAttr.get context ("insert_" ^ input_name))
      (FunctionType.get
         context
         ([ (input_type :> Type.t) ]
          @ List.init input_type#rank (fun _ -> index_type)
          @ [ element_type ])
         [ input_type ]
       |> TypedAttr.get)
      [ Identifier.get context "llvm.emit_c_interface", Attribute.unit context ]
      location
      ~init:(fun body ->
        let insert =
          Tensor.insert
            (body#argument (body#arguments - 1))
            (body#argument 0)
            (List.init (body#arguments - 2) (fun i -> body#argument (i + 1)))
            location
        in
        body#append_operation insert;
        [ insert#result 0 ])
  
  let generate_print input_name input_type =
    Func.func
      context
      (StringAttr.get context ("print_" ^ input_name))
      (FunctionType.get
         context
         [ (input_type :> Type.t) ]
         []
       |> TypedAttr.get)
      [ Identifier.get context "llvm.emit_c_interface", Attribute.unit context ]
      location
      ~init:(fun body ->
        let print = match input_type#encoding with
        | Some _ -> 
          OpBuilder.get "sparse_tensor.print" location
          |> OpBuilder.add_operands [ body#argument 0 ]
          |> OpBuilder.build true
        | None ->
          let zero = Arith.constant (IntegerAttr.get index_type 0) location context in
          body#append_operation zero;
          let result_type = List.init
            input_type#rank
            (fun i -> match input_type#dimension_size i with
            | ShapedType.Static size -> Int.to_string size
            | ShapedType.Dynamic -> "0" (* error *))
            |> String.concat "x"
            |> Printf.sprintf "vector<%sxf64>"
            |> Type.parse context in
          let mask = Arith.constant (FloatAttr.get element_type 0.0 location) location context in
          body#append_operation mask;
          let operandSegmentSizes = [ Int32.one; Int32.of_int input_type#rank; Int32.one; Int32.zero ] in
          let to_vector = OpBuilder.get "vector.transfer_read" location
            |> OpBuilder.add_operands [ body#argument 0 ]
            |> OpBuilder.add_operands (List.init input_type#rank (fun _ -> zero#result 0))
            |> OpBuilder.add_operands [ mask#result 0 ]
            |> OpBuilder.add_results [ result_type ]
            |> OpBuilder.add_attributes [ Identifier.get context "operandSegmentSizes", DenseInt32ArrayAttr.get context operandSegmentSizes ]
            |> OpBuilder.add_attributes [ Identifier.get context "in_bounds", List.init input_type#rank (fun _ -> Attribute.parse context "true") |> ArrayAttr.get context ]
            |> OpBuilder.add_attributes [ Identifier.get context "permutation_map", AffineMap.multi_dim_identity context input_type#rank |> AffineMapAttr.get ]
            |> OpBuilder.build true
          in
          body#append_operation to_vector;
          OpBuilder.get "vector.print" location
          |> OpBuilder.add_operands [ to_vector#result 0 ]
          |> OpBuilder.build true
        in
        body#append_operation print;
        [])

  let generate ast =
    let tensors = Einsum.tensors ast
    and kernels = Einsum.assignments ast in
    let inputs, outputs = Einsum.inputs_and_outputs kernels in
    let module_ = with_module ~init:(fun body ->
      List.iter
        (fun input ->
           let input_type = StringMap.find input tensors in
           generate_init input input_type |> body#append_operation;
           generate_insert input input_type |> body#append_operation)
           (* generate_print input input_type |> body#append_operation) *)
        inputs;
      List.iter
        (fun output ->
           let output_type = StringMap.find output tensors in
           if Option.is_none output_type#encoding
           then generate_init output output_type |> body#append_operation
           else ();
           generate_insert output output_type |> body#append_operation)
           (* generate_print output output_type |> body#append_operation) *)
        outputs) in
    let pass_manager = PassManager.get context in
    let _ = pass_manager#as_op_pass_manager#add_pipeline
      "sparse-assembler{direct-out},sparsification-and-bufferization,sparse-storage-specifier-to-llvm,convert-linalg-to-loops,convert-vector-to-scf,expand-realloc,convert-scf-to-cf,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,canonicalize,cse" in
    let _ = pass_manager#run_on_op module_#to_operation in
    module_
end
