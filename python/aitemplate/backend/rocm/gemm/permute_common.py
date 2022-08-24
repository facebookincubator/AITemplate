# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import jinja2

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const int64_t stride_a = *a_dim1;
{{indent}}const int64_t stride_b = *b_dim1;
{{indent}}const int64_t stride_c = *c_dim1;
    ck::index_t M0 = M / G1 / G2;
    ck::index_t M1 = G1;
    ck::index_t M2 = G2;
    ck::index_t N0 = G3;
    ck::index_t N1 = N / G3;
    // GEMM shape
    //ck::index_t M = M0 * M1 * M2;
    //ck::index_t N = N0 * N1;
    //ck::index_t K = 128;
    //ck::index_t stride_A = K;
    //ck::index_t stride_B = K;
    // E = [M0, N0, M1, N1, M2]
    /* 0, 3, 1, 4, 2
    ck::index_t stride_E_M0 = N0 * M1 * N1 * M2;
    ck::index_t stride_E_M1 = N1 * M2;
    ck::index_t stride_E_M2 = 1;
    ck::index_t stride_E_N0 = M1 * N1 * M2;
    ck::index_t stride_E_N1 = M2;
    */
    // E = [M2, M0, N0, M1, N1] 2, 0, 3, 1, 4
    ck::index_t stride_E_M0 = N0* M1* N1;
    ck::index_t stride_E_M1 = N1;
    ck::index_t stride_E_M2 = M0* N0* M1* N1;
    ck::index_t stride_E_N0 = M1 * N1;
    ck::index_t stride_E_N1 = 1;
    // D = [0, N0, 0, N1, 0]
    ck::index_t stride_D_M0 = 0;
    ck::index_t stride_D_M1 = 0;
    ck::index_t stride_D_M2 = 0;
    ck::index_t stride_D_N0 = N1;
    ck::index_t stride_D_N1 = 1;
"""
)

EXTRA_SHAPE_TEMPLATE_M2N3 = jinja2.Template(
    """
    ck::index_t M0 = M / G1;
    ck::index_t M1 = G1;
    ck::index_t N0 = G2;
    ck::index_t N1 = G3;
    ck::index_t N2 = N / G2 / G3;

    ck::index_t K0 = K;
    ck::index_t G = 1;

    // A[G, M0, M1, M2, K0]
    std::vector<ck::index_t> a_ms_ks_lengths{G, M0, M1, K0};
    std::vector<ck::index_t> a_ms_ks_strides{M0*M1*K0, M1 * K0, K0, 1};
    // B[G, N0, N1, K0]
    std::vector<ck::index_t> b_ns_ks_lengths{G, N0, N1, N2, K0};
    std::vector<ck::index_t> b_ns_ks_strides{N0*N1*N2*K0, N1 * N2 * K0, N2 * K0, K0, 1};

    // D[G, N0, M0, N1, M1, N2]
    std::vector<ck::index_t> d_ms_ns_lengths{G, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> d_ms_ns_strides{N0 * N1 * N2, 0, 0, N1 * N2, N2, 1};

    // E[G, N0, M0, N1, M1, N2] 2, 0, 3, 1, 4
    std::vector<ck::index_t> e_ms_ns_lengths{G, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> e_ms_ns_strides{M0* M1* N0* N1* N2,
                                               N1 * M1 * N2,
                                               N2,
                                               M0 * N1 * M1 * N2,
                                               M1 * N2,
                                               1};

"""
)


EXTRA_SHAPE_TEMPLATE_M3N2 = jinja2.Template(
    """
    ck::index_t M0 = M / G1 / G2;
    ck::index_t M1 = G1;
    ck::index_t M2 = G2;
    ck::index_t N0 = G3;
    ck::index_t N1 = N / G3;

    ck::index_t K0 = K;
    ck::index_t G = 1;


    // A[M0, M1, M2, K0]
    std::vector<ck::index_t> a_ms_ks_lengths{G, M0, M1, M2, K0};
    std::vector<ck::index_t> a_ms_ks_strides{M0 * M1 * M2 * K0, M1 * M2 * K0, M2 * K0, K0, 1};
    // B[N0, N1, K0]
    std::vector<ck::index_t> b_ns_ks_lengths{G, N0, N1, K0};
    std::vector<ck::index_t> b_ns_ks_strides{N0 * N1 * K0, N1 * K0, K0, 1};

    // D[M0, N0, M1, N1, M2]
    std::vector<ck::index_t> d_ms_ns_lengths{G, M0, M1, M2, N0, N1};
    std::vector<ck::index_t> d_ms_ns_strides{N0*N1, 0, 0, 0, N1, 1};
    // E[M0, N0, M1, N1, M2]
    std::vector<ck::index_t> e_ms_ns_lengths{G, M0, M1, M2, N0, N1};
    std::vector<ck::index_t> e_ms_ns_strides{M0 * M1* M2 * N1* N0, N0* M1* N1, N1, M0* N0* M1* N1, M1 * N1, 1};


"""
)
