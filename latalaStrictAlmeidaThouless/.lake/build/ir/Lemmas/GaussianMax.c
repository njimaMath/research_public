// Lean compiler output
// Module: Lemmas.GaussianMax
// Imports: public import Init public meta import Init public import SpinGlass.AT.Gaussian_concentration.gaussian_concentration public import Mathlib.Analysis.Convex.Integral
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
lean_object* initialize_LatalaMeetsAT_SpinGlass_AT_Gaussian__concentration_gaussian__concentration(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Convex_Integral(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LatalaMeetsAT_Lemmas_GaussianMax(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LatalaMeetsAT_SpinGlass_AT_Gaussian__concentration_gaussian__concentration(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Convex_Integral(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
