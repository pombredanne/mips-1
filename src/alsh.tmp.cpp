#include "alsh.tmp.h"


void main_alsh() {
    //todo
    throw std::runtime_error("NotImplementedException");
}

IndexALSH::IndexALSH(size_t L, size_t K, size_t d, float r, float U, size_t m) :
        L_(L), K_(K), d_(d), r_(r), U_(U), m_(m),
        hashtables_(L), a_vectors_(L), b_scalars_(L) {

    reset();
}

void IndexALSH::train(idx_t n, const float* data) {
    train_data_ = from_dtype<float>(data, n, d_);

    std::vector<float> vector_norms(train_data_.vector_count());
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        vector_norms[i] = euclidean_norm(train_data_.row(i), train_data_.vector_length);
    }

    expand(train_data_, vector_norms, false);

    #pragma omp parallel for
    for (size_t l = 0; l < L_; l++) {

        for (size_t i = 0; i < train_data_.vector_count(); i++) {
            std::vector<size_t > hash_vector(K_);

            for (size_t k = 0; k < K_; k++) {
                hash_vector[k] = (size_t) dot_product_hash(a_vectors_[l].row(k), train_data_.row(i), b_scalars_[l][k],
                                                           r_, d_);
            }

            hashtables_[l][hash_vector].insert(i);
        }
    }
}

void IndexALSH::reset() {
    // Initialize random data for all L hash tables
    for (size_t l = 0; l < L_; l++) {
        a_vectors_[l].resize(K_, d_ + 2*m_);
        b_scalars_[l].resize(K_);

        // Generate different random data for each time a vector is hashed
        for (size_t k = 0; k < K_; k++) {
            for (size_t i = 0; i < d_ + 2*m_; i++) {
                a_vectors_[l].at(k, i) = randn();
            }

            auto b = uniform(0, r_);
            b_scalars_[l][k] = b;
        }
    }
}


void IndexALSH::search(idx_t n, const float *data, idx_t k, float *distances, idx_t *labels) {
    auto query_data = from_dtype<float>(data, n, d_);
    std::vector<float> query_norm(query_data.vector_count());
    expand(query_data, query_norm, true);

    #pragma omp parallel for
    for (size_t i = 0; i < query_data.vector_count(); i++) {
        std::map<size_t, size_t> score;

        for (size_t l = 0; l < L_; l++) {

            std::vector<size_t> hash_vector(K_);
            for (size_t k = 0; k < K_; k++) {
                hash_vector[k] = (size_t) dot_product_hash(a_vectors_[l].row(k), query_data.row(i), b_scalars_[l][k],
                                                           r_, d_);
            }

            auto &current_hash_table = hashtables_[l];
            auto &current_bucket = current_hash_table[hash_vector];

            for (auto &it: current_bucket) {
                score[it]++;
            }
        }

        std::vector<std::pair<int, int> > score_vector(score.begin(), score.end());

        sort(score_vector.begin(), score_vector.end(), sort_pred());
        size_t T_count = 0;
    }
}


void IndexALSH::expand(FloatMatrix& data, std::vector<float>& norms, bool queries) {

    auto maximum_norm = max_value(norms);

    #pragma omp parallel for
    for (size_t i = 0; i < data.vector_count(); i++) {
        scale(data.row(i), U_ / maximum_norm, data.vector_length);

        float vec_norm = euclidean_norm(data.row(i), d_);
        norms[i] = vec_norm;

        for (auto j = d_; j < d_ + m_; j++) {
            if (!queries) {
                data.at(i, j) = vec_norm;
                vec_norm *= vec_norm;
            } else {
                data.at(i, j) = 0.5f;
            }
        }

        for (auto j = d_ + m_; j < d_ + 2*m_; j++) {
            if (!queries) {
                data.at(i, j) = 0.5f;
            } else {
                data.at(i, j) = vec_norm;
                vec_norm *= vec_norm;
            }
        }
    }
}
