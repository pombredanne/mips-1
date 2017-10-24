#include "../src/quantization.h"
#include "../src/kmeans.h"
#include "util.wrap.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(mips, m) {
    m.doc() = "MIPS library";

    py::class_<IndexHierarchicKmeans> hkm(m, "IndexHKM");
    py::class_<IndexSubspaceQuantization> sq(m, "IndexSQ");

    WRAP_INDEX_HELPER(IndexHierarchicKmeans, hkm);
    WRAP_INDEX_HELPER(IndexSubspaceQuantization, sq);
}
