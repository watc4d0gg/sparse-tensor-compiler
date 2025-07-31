open Mlir
open Mlir.Ir

module type Environment = sig
  (* val shared_libs : string list *)
  (* val compile : Module.t -> unit *)
  val type_id : TypeId.t
  val dialect_registry : DialectRegistry.t
  val context : Context.t
  val location : Location.t
  val element_type : Type.t
  val index_type : Type.t
  val with_module : init:(Block.t -> unit) -> Module.t
end

(* TODO: narrow down the number of loaded dialects into memory *)
module GlobalEnvironment : Environment = struct
  let type_id =
    let alloc = Mlir.TypeIdAllocator.get () in
    alloc#allocate

  let dialect_registry =
    let registry = DialectRegistry.get () in
    register_all_dialects registry;
    registry

  let context =
    let cont = Context.get (Some dialect_registry) true in
    cont#load_all_available_dialects;
    Mlir.register_all_llvm_translations cont;
    cont

  let location = Location.unknown context
  let element_type = Type.float64 context
  let index_type = Type.index context

  let with_module ~init =
    let m = Module.empty location in
    init m#body;
    assert m#to_operation#verify;
    m
end
