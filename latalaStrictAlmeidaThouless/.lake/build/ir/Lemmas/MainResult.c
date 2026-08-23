// Lean compiler output
// Module: Lemmas.MainResult
// Imports: public import Init public meta import Init public import Lemmas.ATDefs public import Lemmas.Cavity.Talagrand_Cavity public import Lemmas.weak_concentration public import Lemmas.smart_path.proof public import Lemmas.smart_path.mainresult_latala
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_LatalaMeetsAT_Lemmas_ATDefs(uint8_t builtin);
lean_object* initialize_LatalaMeetsAT_Lemmas_Cavity_Talagrand__Cavity(uint8_t builtin);
lean_object* initialize_LatalaMeetsAT_Lemmas_weak__concentration(uint8_t builtin);
lean_object* initialize_LatalaMeetsAT_Lemmas_smart__path_proof(uint8_t builtin);
lean_object* initialize_LatalaMeetsAT_Lemmas_smart__path_mainresult__latala(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LatalaMeetsAT_Lemmas_MainResult(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_Lemmas_ATDefs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_Lemmas_Cavity_Talagrand__Cavity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_Lemmas_weak__concentration(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_Lemmas_smart__path_proof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_Lemmas_smart__path_mainresult__latala(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
