open Mlir
open Core

exception CodegenError of string

module Codegen (Env : Environment) = struct
  module Einsum = Einsum (Env)

  type storage_layout =
    { dims : int list
    ; lvls : int
    ; pos_arrays : int option list
    ; crd_arrays : int option list
    }

  let header_preamble =
    "#include <stdlib.h>\n\
     #include <stdint.h>\n\n\
     typedef struct {\n\
    \    intptr_t    lvls;\n\
    \    intptr_t*   posSizes;\n\
    \    intptr_t**  posArrays;\n\
    \    intptr_t*   crdSizes;\n\
    \    intptr_t**  crdArrays;\n\
    \    intptr_t    numVals;\n\
    \    double*     vals;\n\
     } sparse_tensor_t; \n\n\
     void freeSparseTensor(sparse_tensor_t*);\n\
     void printSparseTensor(sparse_tensor_t*);\n"

  let impl_preamble =
    "#include <stdlib.h>\n\
     #include <stdint.h>\n\
     #include <cstring>\n\
     #include <iostream>\n\
     #include <stdexcept>\n\n\
     template<typename T, size_t N>\n\
     struct MemRefDescriptor {\n\
    \    T* allocated;\n\
    \    T* aligned;\n\
    \    intptr_t offset;\n\
    \    intptr_t sizes[N];\n\
    \    intptr_t strides[N];\n\
     };\n\n\
     template<typename T>\n\
     MemRefDescriptor<T, 1>* arrayToMemRef(intptr_t size, T* array) {\n\
    \    MemRefDescriptor<T, 1>* result = (MemRefDescriptor<T, 1>*) \
     malloc(sizeof(MemRefDescriptor<T, 1>));\n\
    \    result->allocated = array;\n\
    \    result->aligned = array;\n\
    \    result->offset = static_cast<intptr_t>(0);\n\
    \    result->sizes[0] = size;\n\
    \    result->strides[0] = static_cast<intptr_t>(1);\n\
    \    return result;\n\
     }\n\n\
     typedef struct {\n\
    \    intptr_t    lvls;\n\
    \    intptr_t*   posSizes;\n\
    \    intptr_t**  posArrays;\n\
    \    intptr_t*   crdSizes;\n\
    \    intptr_t**  crdArrays;\n\
    \    intptr_t    numVals;\n\
    \    double*     vals;\n\
     } sparse_tensor_t;\n\n\
     extern \"C\" void freeSparseTensor(sparse_tensor_t*);\n\
     extern \"C\" void printSparseTensor(sparse_tensor_t*);\n\n\
     void freeSparseTensor(sparse_tensor_t* tensor) {\n\
    \    free(tensor->posSizes);\n\
    \    free(tensor->crdSizes);\n\
    \    for (int i = 0; i < tensor->lvls; i++) {\n\
    \        free(tensor->posArrays[i]);\n\
    \        free(tensor->crdArrays[i]);\n\
    \    }\n\
    \    // free(tensor->vals);\n\
    \    free(tensor);\n\
     }\n\n\
     void printSparseTensor(sparse_tensor_t* tensor) {\n\
    \    std::cout << \"---- Sparse Tensor ----\" << std::endl;\n\
    \    for (auto dim = 0; dim < tensor->dims; dim++) {\n\
    \        // print pos\n\
    \        auto posSize = tensor->posSizes[dim];\n\
    \        if (posSize > 0) {\n\
    \            std::cout << \"pos[\" << dim << \"] : (\";\n\
    \            auto posArray = tensor->posArrays[dim];\n\
    \            for (auto pos = 0; pos < posSize; pos++) {\n\
    \                std::cout << \" \" << posArray[pos];\n\
    \            }\n\
    \            std::cout << \" )\" << std::endl;\n\
    \        }\n\n\
    \        // print crd\n\
    \        auto crdSize = tensor->crdSizes[dim];\n\
    \        if (crdSize > 0)  {\n\
    \            std::cout << \"crd[\" << dim << \"] : (\";\n\
    \            auto crdArray = tensor->crdArrays[dim];\n\
    \            for (auto crd = 0; crd < crdSize; crd++) {\n\
    \                std::cout << \" \" << crdArray[crd];\n\
    \            }\n\
    \            std::cout << \" )\" << std::endl;\n\
    \        }\n\
    \    }\n\n\
    \    // print vals\n\
    \    std::cout << \"values : (\";\n\
    \    for (auto val = 0; val < tensor->numVals; val++) {\n\
    \        std::cout << \" \" << tensor->vals[val];\n\
    \    }\n\
    \    std::cout << \" )\" << std::endl;\n\
     }\n"

  let indent indent str =
    String.split_on_char '\n' str
    |> List.map (( ^ ) (String.make indent ' '))
    |> String.concat "\n"

  let pos_type size = Printf.sprintf "MemRefDescriptor<intptr_t, %d>" (if size > 1 then 2 else 1)
  let crd_type size = Printf.sprintf "MemRefDescriptor<intptr_t, %d>" (if size > 1 then 2 else 1)
  let vals_type = Printf.sprintf "MemRefDescriptor<double, %d>"

  let storage_layout_type tensor { dims; lvls; pos_arrays; crd_arrays } =
    if List.is_empty pos_arrays && List.is_empty crd_arrays
    then
      Printf.sprintf
        "typedef MemRefDescriptor<double, %d> StorageLayout_%s;\n"
        (List.length dims)
        tensor
    else (
      let fields =
        List.init lvls (fun i ->
          let level_fields = ref [] in
          (match List.nth pos_arrays i with
           | Some size ->
             level_fields := (pos_type size ^ Printf.sprintf " pos%d;" i) :: !level_fields
           | None -> ());
          (match List.nth crd_arrays i with
           | Some size ->
             level_fields := (crd_type size ^ Printf.sprintf " crd%d;" i) :: !level_fields
           | None -> ());
          List.rev !level_fields)
        |> List.flatten
        |> ( @ ) [ vals_type 1 ^ " vals;" ]
      in
      Printf.sprintf
        "typedef struct {\n%s\n} StorageLayout_%s;\n"
        (String.concat "\n" fields |> indent 4)
        tensor)

  let mlir_interfaces tensor { dims; lvls; pos_arrays; crd_arrays } =
    let init =
      Printf.sprintf "void _mlir_ciface_initialize_%s(StorageLayout_%s*);" tensor tensor
    in
    let dissassembled_storage_types =
      List.init lvls (fun i ->
        let level_fields = ref [] in
        (match List.nth pos_arrays i with
         | Some size -> level_fields := (pos_type size ^ "*") :: !level_fields
         | None -> ());
        (match List.nth crd_arrays i with
         | Some size -> level_fields := (crd_type size ^ "*") :: !level_fields
         | None -> ());
        List.rev !level_fields)
      |> List.flatten
      |> ( @ ) [ vals_type (if List.is_empty pos_arrays && List.is_empty crd_arrays then (List.length dims) else 1) ^ "*" ]
      |> String.concat ", "
    in
    let indexing_args = List.init (List.length dims) (fun _ -> "intptr_t") |> String.concat ", " in
    let insert =
      Printf.sprintf
        "void _mlir_ciface_insert_value_%s(StorageLayout_%s*, %s, %s, double);"
        tensor
        tensor
        dissassembled_storage_types
        indexing_args
    in
    Printf.sprintf
      "extern \"C\" {\n%s\n}"
      (String.concat "\n" [ init; insert ] |> indent 4)

  let header_interfaces tensor { dims; _ } extern =
    let init = Printf.sprintf "sparse_tensor_t* init_%s();" tensor in
    let indexing_args = List.init (List.length dims) (fun _ -> "int") |> String.concat ", " in
    let insert =
      Printf.sprintf
        "void insert_%s(sparse_tensor_t* %s, %s, double);"
        tensor
        tensor
        indexing_args
    in
    String.concat "\n" [ (if extern then "extern \"C\" " else "") ^ init; (if extern then "extern \"C\" " else "") ^ insert ]

  let header_implementations tensor { dims; lvls; pos_arrays; crd_arrays } =
    let init =
      let body = ref [] in

      (* allocate the output storage layout *)
      body := Printf.sprintf "StorageLayout_%s* layout = (StorageLayout_%s*) malloc(sizeof(StorageLayout_%s));" tensor tensor tensor :: !body;
      (* call MLIR interface *)
      body := Printf.sprintf "_mlir_ciface_initialize_%s(layout);" tensor :: !body;

      (* newline *)
      body := "" :: !body;

      (* allocate a new sparse tensor *)
      body := "sparse_tensor_t* result = (sparse_tensor_t*) malloc(sizeof(sparse_tensor_t));" :: !body;
      (* set the lvls field *)
      body := Printf.sprintf "result->lvls = static_cast<intptr_t>(%d);" lvls :: !body;

      (* allocate pos sizes and arrays *)
      body := Printf.sprintf "result->posSizes = (intptr_t*) calloc(sizeof(intptr_t), %d);" lvls :: !body;
      body := Printf.sprintf "result->posArrays = (intptr_t**) calloc(sizeof(intptr_t*), %d);" lvls :: !body;
      (* initialize pos sizes and arrays *)
      List.iteri
        (fun i -> function
        | Some size ->
          (* initialize i-th pos size *)
          body := Printf.sprintf "result->posSizes[%d] = layout->pos%d.sizes[0]%s;" i i (if size > 1 then Printf.sprintf " * layout->pos%d.sizes[1]" i else "") :: !body;
          (* initialize i-th pos array data *)
          body := Printf.sprintf "result->posArrays[%d] = layout->pos%d.aligned;" i i :: !body;
        | None -> ())
        pos_arrays;

      (* newline *)
      body := "" :: !body;

      (* allocate crd sizes and arrays *)
      body := Printf.sprintf "result->crdSizes = (intptr_t*) calloc(sizeof(intptr_t), %d);" lvls :: !body;
      body := Printf.sprintf "result->crdArrays = (intptr_t**) calloc(sizeof(intptr_t*), %d);" lvls :: !body;
      (* initialize crd arrays and sizes *)
      List.iteri
        (fun i -> function
        | Some size ->
          (* initialize i-th crd size *)
          body := Printf.sprintf "result->crdSizes[%d] = layout->crd%d.sizes[0]%s;" i i (if size > 1 then Printf.sprintf " * layout->crd%d.sizes[1]" i else "") :: !body;
          (* initialize i-th crd array data *)
          body := Printf.sprintf "result->crdArrays[%d] = layout->crd%d.aligned;" i i :: !body;
        | None -> ())
        crd_arrays;

      (* newline *)
      body := "" :: !body;
      
      (* initialize values *)
      if List.is_empty pos_arrays && List.is_empty crd_arrays then
        (* the tensor is dense *)
        (body := Printf.sprintf "result->numVals = %s;" (List.init (List.length dims) (fun i -> Printf.sprintf "layout->sizes[%d]" i) |> String.concat " * ") :: !body;
        body := "result->vals = layout->aligned;" :: !body)
      else
        (* the tensor is sparse *)
        (body := "result->numVals = layout->vals.sizes[0];" :: !body;
        body := "result->vals = layout->vals.aligned;" :: !body);

      (* newline *)
      body := "" :: !body;

      (* dealloc the output layout *)
      body := "free(layout)" :: !body;
      Printf.sprintf "void init_%s() {\n%s\n}" tensor (List.rev !body |> String.concat "\n" |> indent 4) in

    let insert =
      let indexing_args = List.init (List.length dims) (Printf.sprintf "i%d") in
      
      let body = ref [] in

      (* allocate the output storage layout *)
      body := Printf.sprintf "StorageLayout_%s* layout = (StorageLayout_%s*) malloc(sizeof(StorageLayout_%s));" tensor tensor tensor :: !body;
      (* allocate the tensor's dissassembled storage layout *)
      let pointers =
        List.init lvls (fun i ->
          let level_pointers = ref [] in
          (match List.nth pos_arrays i with
          | Some size ->
            let name = Printf.sprintf "pos%d" i in
            let pos_type = pos_type size in
            (* allocate pos pointer *)
            body := Printf.sprintf "%s* %s = (%s*) malloc(sizeof(%s));" pos_type name pos_type pos_type :: !body;
            (* insert data to the pointer *)
            body := Printf.sprintf "%s->allocated = %s->posArray[%d];" name tensor i :: !body;
            body := Printf.sprintf "%s->aligned = %s->posArray[%d];" name tensor i :: !body;
            body := Printf.sprintf "%s->offset = static_cast<intptr_t>(0);" name :: !body;
            if size > 1 then
              (body := Printf.sprintf "%s->sizes[0] = static_cast<intptr_t>(%s->posSizes[%d] / %d);" name tensor i size :: !body;
              body := Printf.sprintf "%s->sizes[1] = static_cast<intptr_t>(%d);" name size :: !body;
              body := Printf.sprintf "%s->strides[0] = static_cast<intptr_t>(%d);" name size :: !body;
              body := Printf.sprintf "%s->strides[1] = static_cast<intptr_t>(1);" name :: !body;)
            else
              (body := Printf.sprintf "%s->sizes[0] = %s->posSizes[%d];" name tensor i :: !body;
              body := Printf.sprintf "%s->strides[0] = static_cast<intptr_t>(1);" name :: !body);
            level_pointers := name :: !level_pointers
          | None -> ());
          (match List.nth crd_arrays i with
          | Some size ->
            let name = Printf.sprintf "crd%d" i in
            let crd_type = crd_type size in
            (* allocate crd pointer *)
            body := Printf.sprintf "%s* %s = (%s*) malloc(sizeof(%s));" crd_type name crd_type crd_type :: !body;
            (* insert data to the pointer *)
            body := Printf.sprintf "%s->allocated = %s->crdArray[%d];" name tensor i :: !body;
            body := Printf.sprintf "%s->aligned = %s->crdArray[%d];" name tensor i :: !body;
            body := Printf.sprintf "%s->offset = static_cast<intptr_t>(0);" name :: !body;
            if size > 1 then
              (body := Printf.sprintf "%s->sizes[0] = static_cast<intptr_t>(%s->crdSizes[%d] / %d);" name tensor i size :: !body;
              body := Printf.sprintf "%s->sizes[1] = static_cast<intptr_t>(%d);" name size :: !body;
              body := Printf.sprintf "%s->strides[0] = static_cast<intptr_t>(%d);" name size :: !body;
              body := Printf.sprintf "%s->strides[1] = static_cast<intptr_t>(1);" name :: !body;)
            else
              (body := Printf.sprintf "%s->sizes[0] = %s->crdSizes[%d];" name tensor i :: !body;
              body := Printf.sprintf "%s->strides[0] = static_cast<intptr_t>(1);" name :: !body);
            level_pointers := name :: !level_pointers
          | None -> ());
          List.rev !level_pointers)
        |> List.flatten
        |> ( @ ) [
          (* allocate vals pointer *)
          let name = "vals" in
          let is_dense = List.is_empty pos_arrays && List.is_empty crd_arrays in
          let vals_type = vals_type (if is_dense then (List.length dims) else 1) in
          body := Printf.sprintf "%s* %s = (%s*) malloc(sizeof(%s));" vals_type name vals_type vals_type :: !body;
          (* insert data into the pointer *)
          body := Printf.sprintf "%s->allocated = %s->vals;" name tensor :: !body;
          body := Printf.sprintf "%s->aligned = %s->vals;" name tensor :: !body;
          if is_dense then
            let strides = List.fold_left (fun curr dim -> (dim * List.hd curr) :: curr) [1] (List.drop 1 dims |> List.rev) in
            List.iter2
              (fun i dim -> )
              dims
          else

          name
          ]
      in
      (* call MLIR interface *)
      body := Printf.sprintf "_mlir_ciface_insert_value_%s(layout);" tensor :: !body;

      (* newline *)
      body := "" :: !body;

      (* allocate a new sparse tensor *)
      body := "sparse_tensor_t* result = (sparse_tensor_t*) malloc(sizeof(sparse_tensor_t));" :: !body;
      (* set the lvls field *)
      body := Printf.sprintf "result->lvls = static_cast<intptr_t>(%d);" lvls :: !body;

      (* allocate pos sizes and arrays *)
      body := Printf.sprintf "result->posSizes = (intptr_t*) calloc(sizeof(intptr_t), %d);" lvls :: !body;
      body := Printf.sprintf "result->posArrays = (intptr_t**) calloc(sizeof(intptr_t*), %d);" lvls :: !body;
      (* initialize pos sizes and arrays *)
      List.iteri
        (fun i -> function
        | Some size ->
          (* initialize i-th pos size *)
          body := Printf.sprintf "result->posSizes[%d] = layout->pos%d.sizes[0]%s;" i i (if size > 1 then Printf.sprintf " * layout->pos%d.sizes[1]" i else "") :: !body;
          (* initialize i-th pos array data *)
          body := Printf.sprintf "result->posArrays[%d] = layout->pos%d.aligned;" i i :: !body;
        | None -> ())
        pos_arrays;

      (* newline *)
      body := "" :: !body;

      (* allocate crd sizes and arrays *)
      body := Printf.sprintf "result->crdSizes = (intptr_t*) calloc(sizeof(intptr_t), %d);" lvls :: !body;
      body := Printf.sprintf "result->crdArrays = (intptr_t**) calloc(sizeof(intptr_t*), %d);" lvls :: !body;
      (* initialize crd arrays and sizes *)
      List.iteri
        (fun i -> function
        | Some size ->
          (* initialize i-th crd size *)
          body := Printf.sprintf "result->crdSizes[%d] = layout->crd%d.sizes[0]%s;" i i (if size > 1 then Printf.sprintf " * layout->crd%d.sizes[1]" i else "") :: !body;
          (* initialize i-th crd array data *)
          body := Printf.sprintf "result->crdArrays[%d] = layout->crd%d.aligned;" i i :: !body;
        | None -> ())
        crd_arrays;

      (* newline *)
      body := "" :: !body;
      
      (* initialize values *)
      if List.is_empty pos_arrays && List.is_empty crd_arrays then
        (* the tensor is dense *)
        (body := Printf.sprintf "result->numVals = %s;" (List.init dims (fun i -> Printf.sprintf "layout->sizes[%d]" i) |> String.concat " * ") :: !body;
        body := "result->vals = layout->aligned;" :: !body)
      else
        (* the tensor is sparse *)
        (body := "result->numVals = layout->vals.sizes[0];" :: !body;
        body := "result->vals = layout->vals.aligned;" :: !body);

      (* newline *)
      body := "" :: !body;

      (* dealloc the output layout *)
      body := "free(layout)" :: !body;
      Printf.sprintf "void insert_%s(sparse_tensor_t* %s, %s, double value) {\n%s\n}" tensor tensor (List.map (fun arg -> "int " ^ arg) indexing_args |> String.concat ", ") (List.rev !body |> String.concat "\n" |> indent 4) in
    init

  let emit ast header_out impl_out =
    Printf.fprintf header_out "%s" header_preamble;
    Printf.fprintf impl_out "%s" impl_preamble;
    let tensors = Einsum.tensors ast in
    let kernels = Einsum.assignments ast in
    let inputs, outputs = Einsum.inputs_and_outputs kernels in
    let _dense_outputs =
      List.filter
        (fun output -> (StringMap.find output tensors)#encoding |> Option.is_none)
        outputs
    in
    (* STORAGE LAYOUTS:
      * for each tensor:
      * if sparse:
          * compute how many pos arrays (# of compressed levels)
          * compute how many crd arrays (# of compressed levels)
          * one vals array
          * order them: pos_0, crd_0, pos_1, crd_1,..., vals
          * each array is 1-D
          * pos and crd arrays store intptr_t, vals store doubles
          * define a struct with MemRefDescriptors and typedef it
      * if dense:
          * the storage layout is only a vals array
          * N-D depending on the # of dims
          * typedef the MemRefDescriptor *)
    let storage_layouts =
      List.fold_left
        (fun map tensor ->
           let tensor_type = StringMap.find tensor tensors in
           let tensor_dims = List.init
            tensor_type#rank
            (fun _ -> match tensor_type#dimension_size with
            | ShapedType.Static size -> size
            | ShapedType.Dynamic -> CodegenError "Dynamic dimension sizes are not supported!" |> raise)
          in
           match tensor_type#encoding with
           | Some encoding ->
             let level_types = List.init encoding#level_rank encoding#level_type in
             let pos_arrays, crd_arrays =
               List.mapi
                 (fun i t ->
                    match t with
                    | LevelType.Compressed _ ->
                      (* for COO formats: count the number of COO levels as that
                         will define the size of the second dimension of the crd memref *)
                      let singleton_levels =
                        List.drop (i + 1) level_types
                        |> List.fold_left
                             (fun count -> function
                                | LevelType.Singleton _ -> count + 1
                                | _ -> count)
                             0
                      in
                      Some 1, Some singleton_levels
                    | LevelType.LooseCompressed _ -> Some 2, Some 1
                    | _ -> None, None)
                 level_types
               |> List.split
             in
             StringMap.add
               tensor
               { dims = tensor_dims
               ; lvls = encoding#level_rank
               ; pos_arrays
               ; crd_arrays
               }
               map
           | None ->
             StringMap.add
               tensor
               { dims = tensor_dims
               ; lvls = tensor_type#rank
               ; pos_arrays = []
               ; crd_arrays = []
               }
               map)
        StringMap.empty
        (inputs @ outputs)
    in
    List.iter
      (fun tensor ->
         (* Tensor's storage Layout *)
         let storage_layout = StringMap.find tensor storage_layouts in

         (* HEADER *)
         Printf.fprintf header_out "\n";
         (* Emit header interface function:
            * for each tensor:
              * generate init => no inputs + sparse_tensor_t* result
              * generate insert => inputs: sparse_tensor_t* + indexing coordinates in dim order + a value to be inserted *)
         header_interfaces tensor storage_layout false |> Printf.fprintf header_out "%s\n";

         (* IMPLEMENTATION *)
         Printf.fprintf impl_out "\n";
         (* Emit storage layout typedef *)
         storage_layout_type tensor storage_layout |> Printf.fprintf impl_out "%s\n";
         (* Emit external MLIR interface functions *)
         mlir_interfaces tensor storage_layout |> Printf.fprintf impl_out "%s\n";
         (* Emit C++ header interface implementations *)
         header_interfaces tensor storage_layout true |> Printf.fprintf impl_out "%s\n";
         header_implementations tensor storage_layout |> Printf.fprintf impl_out "%s\n")
      (inputs @ outputs)
      (* generate kernel => inputs: sparse_tensor_t* for each input and output in order (inputs_order + outputs_order) *)
end
