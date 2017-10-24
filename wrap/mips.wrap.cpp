#include "../src/quantization.h"
#include "../src/kmeans.h"
#include "util.wrap.h"

namespace py = pybind11;

using layer_t = IndexHierarchicKmeans::layer_t;

PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<size_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<layer_t>);

PYBIND11_MODULE(mips, m) {
    m.doc() = "MIPS library";

    // FLOAT MATRIX ----------------------------------------------------------------------------------------------------
    py::class_<FloatMatrix> fm(m, "FloatMatrix", py::buffer_protocol());
    fm.def_buffer([](FloatMatrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data.data(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            2,
            { m.vector_count(), m.vector_length },
            { sizeof(float) * m.vector_count(), sizeof(float) }
        );
    });

    // K-MEANS RESULT---------------------------------------------------------------------------------------------------
    py::class_<kmeans_result>(m, "kmeans_result")
        .def_readonly("centroids", &kmeans_result::centroids)
        .def_property_readonly("assignments",
                               [](kmeans_result& self) {
                                   return STLVectorWrapper<size_t>(self.assignments).as_array();},
                               py::keep_alive<0, 1>());

    // LAYER_T ---------------------------------------------------------------------------------------------------------
    py::class_<layer_t>(m, "layer_t")
        .def_readonly("kr", &layer_t::kr)
        .def_readonly("centroid_children", &layer_t::centroid_children)
        .def_readonly("cluster_num", &layer_t::cluster_num);

    // K-MEANS ---------------------------------------------------------------------------------------------------------
    py::class_<IndexHierarchicKmeans> hkm(m, "IndexHKM");
    hkm.def(
        py::init<size_t, size_t, size_t, size_t>(),
        "no docstring",
        py::arg("dim"), py::arg("m"), py::arg("layers_count"), py::arg("opened_trees")
    );
    hkm.def_readonly("layers_count", &IndexHierarchicKmeans::layers_count);
    hkm.def_readonly("m",            &IndexHierarchicKmeans::m);
    hkm.def_readonly("opened_trees", &IndexHierarchicKmeans::opened_trees);
    hkm.def_readonly("vectors",      &IndexHierarchicKmeans::vectors, py::return_value_policy::reference);
    hkm.def_readonly("layers" ,      &IndexHierarchicKmeans::layers,  py::return_value_policy::reference);
    WRAP_INDEX_HELPER(IndexHierarchicKmeans, hkm);

    // QUANTIZATION ----------------------------------------------------------------------------------------------------
    py::class_<IndexSubspaceQuantization> sq(m, "IndexSQ");
    WRAP_INDEX_HELPER(IndexSubspaceQuantization, sq);

}
