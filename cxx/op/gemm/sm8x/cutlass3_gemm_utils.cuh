
#include "cute/tensor.hpp"
#include "utils/ada_fp8_mma_trait.hpp"

namespace LAB{
namespace CUDA{
namespace SM8X{
    
template<typename T>
struct MMATrait{};

template<>
struct MMATrait<cutlass::float_e4m3_t>{
    // 16x16x32
    using MMA_M  = Int<int(M_T{} * 1 * 16)>;
    using MMA_K  = Int<int(K_T{} * 1 * 32)>;
    using MMA_N  = Int<int(N_T{} * 1 * 16)>;
    using Atom = SM89_16x8x32_F32F8F8F32_E4M3_TN;
    using DTypeD = float; 
};


template<>
struct MMATrait<cutlass::float_e5m2_t>{
    // 16x16x32
    using MMA_M  = Int<int(M_T{} * 1 * 16)>;
    using MMA_K  = Int<int(K_T{} * 1 * 32)>;
    using MMA_N  = Int<int(N_T{} * 1 * 16)>;
    using Atom = SM89_16x8x32_F32F8F8F32_E5M2_TN;
    using DTypeD = float; 
};


template<>
struct MMATrait<cutlass::int8_t>{
    // 16x16x32
    using MMA_M  = Int<int(M_T{} * 1 * 16)>;
    using MMA_K  = Int<int(K_T{} * 1 * 32)>;
    using MMA_N  = Int<int(N_T{} * 1 * 16)>;
    using Atom = SM80_16x8x32_S32S8S8S32_TN;
    using DTypeD = int32_t; 
};

template<>
struct MMATrait<cutlass::half_t>{
    // 16x8x16
    using MMA_M  = Int<int(M_T{} * 1 * 16)>;
    using MMA_K  = Int<int(K_T{} * 1 * 32)>;
    using MMA_N  = Int<int(N_T{} * 1 * 16)>;
    using Atom = SM80_16x8x16_F32F16F16F32_TN;
    using DTypeD = int32_t; 
};



}
}   
}