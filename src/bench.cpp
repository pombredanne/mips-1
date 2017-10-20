#include "bench.h"

#include <cstdio>
#include <cassert>

#include <sys/time.h>

#include "faiss/AutoTune.h"
#include "../src/common.h"


double elapsed () {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

#if 0
std::string filenames[4] = {
	"sift1M/sift_learn.fvecs",
	"sift1M/sift_base.fvecs",
	"sift1M/sift_query.fvecs",
	"sift1M/sift_groundtruth.ivecs",
};
#else
std::string filenames[4] = {
	"siftsmall/siftsmall_learn.fvecs",
	"siftsmall/siftsmall_base.fvecs",
	"siftsmall/siftsmall_query.fvecs",
	"siftsmall/siftsmall_groundtruth.ivecs",
};
#endif

int bench(faiss::Index* get_trained_index(const FloatMatrix& xt)) {
    double t0 = elapsed();

    faiss::Index* index;
    size_t d;

    {
        printf ("[%.3f s] Loading train set\n", elapsed() - t0);

        FloatMatrix xt = load_vecs<float>(filenames[0]);
        d = xt.vector_length;

        printf ("[%.3f s] Preparing index d=%zu and training on %zu vectors\n",
                elapsed() - t0, d, xt.vector_count());
        index = get_trained_index(xt);
    }


    {
        printf ("[%.3f s] Loading database\n", elapsed() - t0);

        FloatMatrix xb = load_vecs<float>(filenames[1]);
        size_t d2 = xb.vector_length;
        size_t nb = xb.vector_count();
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);

        index->add(nb, xb.data.data());
    }

    FloatMatrix xq;
    size_t nq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        xq = load_vecs<float>(filenames[2]);
        size_t d2 = xq.vector_length;
        nq = xq.vector_count();
        assert(d == d2 || !"query does not have same dimension as train set");

    }

    size_t k; // nb of results per query in the GT
    FlatMatrix<faiss::Index::idx_t> gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        FlatMatrix<int> gt_int = load_vecs<int>(filenames[3]);
        k = gt_int.vector_length;
        size_t nq2 = gt_int.vector_count();

        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt.resize(nq2, k);
        for(size_t i = 0; i < k * nq; i++) {
            gt.data[i] = gt_int.data[i];
        }
    }


    { // Use the found configuration to perform a search
        printf ("[%.3f s] Perform a search on %ld queries\n",
                elapsed() - t0, nq);

        // output buffers
        faiss::Index::idx_t *I = new faiss::Index::idx_t[nq * k];
        float *D = new float[nq * k];

        index->search(nq, xq.data.data(), k, D, I);

        printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(size_t i = 0; i < nq; i++) {
            int gt_nn = gt.at(i, 0);
            for(size_t j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

    }

    delete index;

    return 0;
}