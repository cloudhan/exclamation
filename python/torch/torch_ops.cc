#include "python/torch/torch_ops.h"

PYBIND11_MODULE(_exclamation_ops, m) {
  m.def("paged_attention", &exclamation::python::paged_attention);
  m.def("lbp_attention", &exclamation::python::lbp_attention);
}
