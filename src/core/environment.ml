open Mlir
open Mlir.Ir

module type Environment = sig
  val dialect_registry : DialectRegistry.t
  val context : Context.t
end

(* TODO: narrow down the number of loaded dialects into memory *)
module GlobalEnvironment : Environment = struct
  let dialect_registry =
    let registry = DialectRegistry.get () in
    register_all_dialects registry;
    registry

  let context =
    let cont = Context.get (Some dialect_registry) true in
    cont#load_all_available_dialects;
    cont
end
