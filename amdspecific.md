The AMD Radeon RX 7900 XT features 5,376 stream processors and 84 Compute Units (CUs) within AMD's RDNA 3 architecture, providing high parallel processing throughput for gaming and compute tasks.  It utilizes a multi-chip Navi 31 design with 20GB of GDDR6 VRAM and a 320-bit memory bus, delivering 800 GB/s of bandwidth to support these parallel operations. 

In AI and machine learning workloads, the card's parallel capabilities are leveraged through ROCm software support, enabling tensor parallelism for large language model inference. While it lacks the dedicated tensor cores found in NVIDIA's architecture, its dual-issue FP32 execution units allow it to dispatch multiple instructions per clock, making it a viable option for budget-conscious local AI setups, particularly when paired with optimized runtimes like vLLM or llama.cpp.

AMD GPUs utilize a compute unit (CU) architecture that supports instruction-level parallelism through a specialized issue arbiter. This arbiter can dispatch up to five instructions per cycle to different execution units, selecting warps in a round-robin fashion to maximize throughput. 

The theoretical maximum five-instruction-per-cycle throughput includes:

One VALU instruction: Handling vector arithmetic and computation. 
One vector memory operation: Managing data movement.
One SALU and/or scalar memory operation: Executing scalar logic or memory accesses.
One LDS operation: Accessing local data share memory.
One branch operation: Handling control flow.
This dual-issuing capability allows the GPU to efficiently overlap arithmetic, memory, and control operations, significantly boosting performance for parallel workloads executed via the HIP programming model.

Hardware implementation
This topic describes the hardware architecture of AMD GPUs supported by HIP, focusing on the internal organization and operation of GPU hardware components. Understanding these hardware details helps you optimize GPU applications and achieve maximum performance.

Overall GPU architecture
AMD GPUs consist of interconnected blocks of digital circuits that work together to execute complex parallel computing tasks. Unlike central processing units (CPUs), which dedicate significant silicon area to instruction flow control, branch prediction, and complex caching hierarchies, GPUs allocate the majority of their die area to arithmetic pipelines. This design choice enables extreme throughput density for data-parallel workloads. The architecture is organized hierarchically to enable massive parallelism while efficiently managing resources.

Command processor and control
The command processor (CP) serves as the primary interface between the CPU and GPU, receiving and distributing commands for execution. The CP consists of two main components:

Command processor fetcher (CPF): Fetches commands from memory and passes them to the command processor packet processor (CPC) for processing.

Command processor packet processor (CPC): A microcontroller that decodes the fetched commands and dispatches kernels to the workgroup processors for scheduling.

The command processor handles several types of operations:

Kernel launches, which are forwarded to asynchronous compute engines (ACEs)

Memory transfers, which are delegated to direct memory access (DMA) engines

Synchronization operations and memory fences

DMA engines handle memory transfers between CPU and GPU memory without CPU involvement after initialization. Most GPUs contain two DMA engines, enabling concurrent bidirectional transfers to better utilize PCIe bandwidth. The DMA engines fetch data in small chunks and can process transfers in parallel but cannot handle multiple copy commands on the same engine simultaneously.

Asynchronous compute engines (ACEs) break down kernels into workgroups for distribution to shader processor input (SPI) blocks. Multiple ACEs enable concurrent kernel execution, with each ACE capable of dispatching one kernel at a time. ACEs process commands from different queues asynchronously, enabling overlap between different kernel executions and memory operations.

Hierarchical organization
The GPU organizes compute resources in a three-level hierarchy that enables modular design and resource sharing:

Shader engines (SE): Top-level organizational units containing multiple shader arrays and shared resources

Shader arrays: Groups of compute units (CUs) sharing instruction and scalar caches

Compute units (CU): Basic execution units containing the arithmetic logic units (ALUs) and registers for thread execution

Diagram showing the hierarchical organization of compute units grouped into shader engines on AMD GPUs
Hierarchical organization of compute units into shader engines
This hierarchical design allows different GPU configurations using the same underlying architecture.

Shader engine components
Shader engines group multiple compute units together, sharing resources to improve efficiency and reduce redundancy. Each shader engine contains several key components shared across its compute units.

Workgroup manager (SPI)
The workgroup manager, also called the shader processor input (SPI), bridges the command processor and compute units. After the CP processes a kernel dispatch, the SPI:

Receives workgroups from the ACEs

Schedules workgroups onto available compute units

Initializes registers with kernel parameters

Ensures all warps of a workgroup execute on the same CU for synchronization

Monitors resource availability and queues workgroups when resources are exhausted

The SPI tracks four critical resources that limit concurrent execution:

warp slots (execution contexts)

Vector general-purpose registers (VGPRs)

Scalar general-purpose registers (SGPRs)

Local data share (LDS) memory

Workgroup-to-CU mapping is non-deterministic and based on available resources. You should not assume any specific mapping pattern, as the same kernel launched multiple times can have different workgroup distributions.

Scalar L1 data cache (sL1D)
The scalar L1 data cache (sL1D) serves scalar memory operations from multiple CUs within a shader array. The sL1D is shared between CUs and caches data that is uniform across a warp, including:

Kernel arguments and pointers

Grid and block dimensions

Constants accessed uniformly across threads

Data from __constant__ memory when accessed uniformly

Unlike the vector L1 cache, the sL1D doesn’t use a “hit-on-miss” approach, meaning subsequent requests to the same pending cache line count as duplicated misses rather than hits.

L1 instruction cache (L1I)
The L1 instruction cache (L1I) is a read-only cache shared between multiple CUs in a shader array. Like the sL1D, it’s backed by the L2 cache and doesn’t use the “hit-on-miss” approach. The L1I stores kernel instructions fetched by the compute units, reducing instruction fetch latency and L2 cache pressure.

Compute unit architecture
The compute unit (CU) is the fundamental execution block of AMD GPUs, serving as the atomic building block for massive parallelism. Each CU is responsible for executing kernels through its various specialized components and pipelines. Data flows into these pipelines, undergoes arithmetic transformation, and exits as results, to maximize the number of such transformations per clock cycle.

CUs enable latency hiding through massive hardware multithreading. A single CU can manage thousands of concurrent threads organized as a number of warps, each containing 32 (RDNA) or 64 (CDNA) threads. This massive concurrency allows the hardware to hide memory access latency by executing other warps while some wait for data.

Detailed diagram of an AMD CDNA compute unit showing internal components and data flow
Internal architecture of an AMD CDNA compute unit
Sequencer and scheduling
The instruction sequencer (SQ) serves as the control center of each compute unit, managing instruction flow through the execution pipelines. The sequencer maintains warp state and coordinates instruction execution across different functional units.

Warp organization: The sequencer organizes active warps into four pools, each containing slots for up to ten warps (eight on the CDNA2 MI200 Series). Each slot includes:

Warp-level registers (program counter, execution mask, and others)

Instruction buffer for prefetched instructions

State information for scheduling decisions

This organization theoretically allows up to 40 concurrent warps per CU, though actual occupancy is typically limited by register and LDS usage.

Instruction fetching: The fetch arbiter selects one warp per cycle to fetch instructions from memory, prioritizing the oldest warps. Each CU can fetch up to 32 bytes (4-8 instructions) per cycle.

Instruction issuing: The issue arbiter determines which instructions execute each cycle, selecting warps from one pool per cycle in round-robin fashion. The arbiter can issue multiple instructions per cycle to different execution units, with a theoretical maximum of five instructions per cycle:

One VALU instruction

One vector memory operation

One SALU and/or scalar memory operation

One LDS operation

One branch operation

Instructions always issue at warp granularity, with all threads in the warp executing the same instruction in lockstep. The hardware can perform single-cycle context switching between warps with zero overhead, as all warp contexts remain resident on the CU. This enables efficient latency hiding, allowing the CU to switch to another warp immediately when the current warp encounters a stall condition such as a memory access.

Execution pipelines
Each CU contains multiple specialized execution pipelines that process different types of instructions in parallel, enabling efficient utilization of the hardware resources.

Vector arithmetic logic unit (VALU)
The VALU executes vector instructions across entire warps, with each thread potentially operating on different data. For CDNA architectures, the VALU consists of:

Four SIMD processors: Each containing 16 single-precision ALUs (or equivalent), for 64 total ALUs per CU. In CDNA3, these are SIMD64 pipelines that can execute 256 operations per cycle per CU.

Vector register files: 256-512 KiB of VGPR storage split across the four SIMDs. VGPRs are organized as 32-bit lanes, providing flexibility for mixed-precision computations.

Instruction buffers: Storage for up to 8-10 warps per SIMD

On architectures with 64-thread warps and 16-instruction wide SIMD units, executing one instruction takes four cycles (one cycle per 16 threads). The four SIMD design ensures full utilization when sufficient warps are available, as a new instruction can issue to each SIMD every cycle.

The VALU serves as the primary arithmetic engine, executing the majority of computation in GPU kernels. Data flows into these pipelines, undergoes arithmetic transformation, and exits as results, with the goal of maximizing the number of such transformations per clock cycle.

For CDNA architectures with matrix operations, the VALU also dispatches matrix fused multiply-add (MFMA) instructions to specialized matrix units.

Register pressure and occupancy
Register usage directly impacts CU occupancy. Each warp requires a portion of the finite VGPR and SGPR pools. Higher register usage per thread reduces the maximum number of concurrent warps, potentially limiting the CU’s ability to hide latency. Mixed-precision workloads can optimize register usage by storing lower-precision values in fewer registers.

Scalar arithmetic logic unit (SALU)
The SALU executes instructions uniformly across all threads in a warp, handling operations such as:

Control flow (branches, loops)

Address calculations

Loading kernel arguments and constants

Managing warp-uniform values

The SALU includes:

A scalar processor for arithmetic and logic operations

12.5 KiB of SGPR storage per CU

A scalar memory (SMEM) unit for memory operations

Scalar operations reduce pressure on vector units and registers by handling uniform computations efficiently.

Vector memory unit (VMEM)
The VMEM unit handles all vector memory operations, including loads, stores, and atomic operations. Each thread supplies its own address and data, though the hardware optimizes access through memory coalescing when threads access nearby addresses. The VMEM unit connects to the vector L1 cache and implements both address generation and coalescing logic.

Branch unit
The branch unit executes jumps and branches for control flow changes affecting entire warps. Note that the branch unit handles warp-level control flow, not execution mask updates for thread divergence, which are handled through predication.

Special function unit (SFU)
The special function units accelerate certain arithmetic operations that are too complex and/or costly to implement purely within the standard vector ALUs.

SFUs are responsible for executing transcendental and reciprocal mathematical functions, operations such as exp, log, sin, cos, rcp (reciprocal), and rsqrt (reciprocal square root). These are heavily used in scientific, physics, and machine learning workloads, particularly in activation functions such as GELU, sigmoid, and/or softmax.

Each CU includes a set of specialized pipelines and/or transcendental function units (TFUs) that handle these operations with dedicated hardware. While their throughput is lower than that of the primary SIMD pipelines, they enable these functions to execute efficiently without consuming general ALU bandwidth.

From the compiler’s perspective, these operations map to specific AMDGPU ISA instructions, such as:

v_exp_f32 - compute exponential base e

v_log_f32 - compute natural logarithm

v_sin_f32, v_cos_f32 - compute sine and/or cosine

v_rsq_f32, v_rcp_f32 - compute reciprocal and/or reciprocal square root

In CDNA3-based GPUs (such as MI300), SFU throughput and latency have been tuned for deep learning primitives. For instance, exponentiation (exp) and logarithm (log) functions are now pipelined to complete in a few cycles per lane, allowing vectorized activation functions in large-scale matrix workloads to execute without significant stalls.

For programmers targeting ROCm and/or HIP, these SFU-accelerated operations are typically accessed through math intrinsics such as __expf, __logf, and/or __sinf, which the compiler lowers to the corresponding AMDGPU ISA instructions at compile time.

Load/store unit (LSU)
The load/store units handle the transfer of data between the compute units and the GPU’s memory subsystems. They are responsible for issuing, tracking, and retiring memory operations, including loads from and stores to global memory, local shared memory, and caches, for thousands of concurrent threads.

Each CU includes a set of LSUs tightly integrated with its vector and scalar pipelines. These units handle memory instructions generated by active warps, such as buffer_load, buffer_store, and flat_load_dword, and route them through the GPU’s hierarchical memory system.

The LSU’s responsibilities include:

Managing vector memory accesses for SIMD instructions

Coordinating local data share (LDS) reads and writes

Accessing the L0 and/or L1 caches and forwarding requests to the L2 cache and high-bandwidth memory (HBM)

Handling synchronization and atomic operations between threads and workgroups

LSUs manage thousands of outstanding memory requests per GPU, dynamically scheduling them to hide memory latency. While arithmetic pipelines continue executing other warps, the LSUs maintain queues of pending transactions and reorder responses as data returns from memory.

Matrix fused multiply-add (MFMA)
CDNA architectures (MI100 and newer) include specialized matrix acceleration units for high-throughput matrix operations. These units execute independently from other VALU operations, allowing overlap between matrix and vector computations. MFMA units support various data types including INT8, FP16, BF16, and FP32, with different throughput characteristics for each.

Matrix cores are GPU execution units that perform large-scale matrix operations in a single instruction. In AMD architectures, these units are formally known as MFMA (matrix fused multiply-add) units, the core hardware blocks responsible for accelerating deep learning, high-performance computing (HPC), and dense linear-algebra workloads on modern Instinct GPUs.

Operating on entire tiles of matrices per instruction allows MFMA units to deliver far greater arithmetic throughput and energy efficiency than scalar and/or vector ALUs. Rather than fetching and decoding thousands of per-element multiply-add instructions, each MFMA instruction processes an entire matrix fragment, drastically reducing power per operation and increasing overall throughput. The MFMA units implement a mini-systolic array design that efficiently processes matrix tiles.

An example MFMA instruction from the AMDGPU ISA is:

v_mfma_f32_16x16x4f16 v[0:15], v[16:31], v[32:47], v[0:15]
This instruction performs a matrix multiplication and accumulation 
, where the fragments 
, 
, and 
 are stored in VGPRs. The suffix 16x16x4f16 indicates a tile size of 
, with an inner dimension of 
, operating on half-precision (FP16) inputs and accumulating into 32-bit floating-point outputs.

Programmers can access MFMA functionality at multiple levels: through optimized libraries, compiler intrinsics, and/or inline assembly, providing flexibility for different use cases.

The MFMA units use both standard VGPRs and additional accumulation VGPRs (AGPRs) on supported architectures, providing up to 512 KiB of combined register storage per CU.

Data movement engine (CDNA 3 / CDNA 4)
CDNA 3 and CDNA 4 architectures include specialized Data Movement Engine (DME) hardware units designed to accelerate access to multi-dimensional tensor data in GPU memory. DMEs perform high-throughput, low-overhead copies between global memory (HBM) and the on-chip memory hierarchy, particularly the Local Data Share (LDS) and L0 and/or L1 caches, without consuming compute resources.

A DME issues bulk memory transactions for contiguous and/or affine data regions, such as tensors laid out as multi-dimensional arrays in global memory. The hardware computes memory addresses for large block transfers in parallel, offloading this work from the SIMD pipelines and reducing pressure on both the register file and the instruction scheduler. This enables higher sustained bandwidth and lower latency for operations involving tiled matrix and/or tensor data.

In practice, DMEs handle transfers of the form 
 across many threads and dimensions simultaneously. By performing these affine address calculations directly in hardware, the DME avoids the need for per-thread address arithmetic, freeing up scalar ALUs and registers for computation.

The DME design provides two key advantages:

Resource decoupling: By removing large tensor copies from the main execution pipelines, the CU can continue executing arithmetic instructions while data movement occurs in the background.

Asynchronous execution model: A single warp can issue a DME copy command, immediately resume computation, and later synchronize only when the transfer has completed. This enables producer-consumer parallelism.

Programmers can access this functionality through asynchronous copy intrinsics in ROCm, such as __builtin_amdgcn_async_work_group_copy, which map directly to hardware-level DME operations. These intrinsics allow explicit control over data transfer overlap, synchronization, and cache placement.

Local data share (LDS)
The local data share provides fast on-CU scratchpad memory for communication between threads in a workgroup.

Diagram showing the organization of local data share with banks and connections to SIMD units
Local data share organization and SIMD connections
Organization: The LDS contains 32 (CDNA, CDNA 2, and CDNA 3) or 64 (CDNA 4 and RDNA 2, RDNA 3, and RDNA 4) banks, each 4-bytes wide, providing 128 (CDNA, CDNA 2, and CDNA 3) or 256 (CDNA 4 and RDNA 2, RDNA 3, and RDNA 4) bytes per cycle total bandwidth. Each bank can be accessed independently every cycle for reads, writes, and/or atomic operations. The SIMDs connect to the LDS in pairs, with each pair sharing a 64-byte bidirectional port.

Access patterns: A single warp can achieve up to 64 bytes per cycle throughput (16 lanes per cycle). The actual bandwidth depends on data size and access patterns:

4-byte values: 8 cycles for 64 threads (50% peak bandwidth)

16-byte values: 20 cycles for 64 threads (80% peak bandwidth)

Conflict resolution: The LDS includes hardware to detect and resolve bank conflicts when multiple threads access different addresses in the same bank. Conflicts are resolved by serializing accesses across multiple cycles. Address conflicts (multiple threads atomically updating the same address) are similarly serialized. Broadcasting from the same address to multiple threads is handled efficiently without conflicts.

Vector L1 cache
Each CU contains a dedicated vector L1 data cache (vL1D) serving vector memory operations. Key characteristics include:

Write-through design (writes go directly to L2)

Optimization for high-bandwidth streaming access patterns

Coherent with other CUs through software management

Typical size of 16 KB per CU

The vector cache tags are checked for all vector memory operations, with misses forwarded to the L2 cache. The write-through design simplifies coherence at the cost of write bandwidth.

Memory hierarchy and system
The GPU memory system provides the bandwidth and capacity needed for massive parallel computation while managing data coherence and access efficiency.

Memory organization
Block diagram showing four compute engines with L2 cache, memory controllers, and Infinity Fabric interconnect on CDNA2
CDNA2 Graphics Compute Die organization showing memory subsystem
AMD GPUs typically use high-bandwidth memory (HBM) for data-intensive workloads, providing significantly higher bandwidth than traditional GDDR memory at the cost of slightly higher latency. HBM achieves this through vertical stacking of memory dies and wide memory buses, enabling massive parallel access to memory channels.

The memory system includes:

Memory channels: Multiple independent memory controllers (typically 8-16)

L2 cache banks: Distributed cache banks serving as the coherence point

Infinity Fabric: High-speed interconnect for data routing

L2 cache architecture
The L2 cache serves as the coherence point for all GPU memory accesses and is shared by all compute units. The L2 consists of multiple independent channels (32 on CDNA GPUs at 256-byte interleaving) that operate in parallel.

Diagram showing L2 cache to Infinity Fabric transaction flow with request categorization and routing
L2 cache to Infinity Fabric transaction flow
Key characteristics:

Channel organization: Each channel handles a portion of the address space, with addresses interleaved across channels for load balancing.

Hit-on-miss behavior: If a request arrives for a pending cache line fill, it counts as a hit, improving the effective hit rate.

Write coalescing: Multiple writes to the same cache line are combined.

Atomic operation support: Atomics execute directly in the L2 cache for coherence.

L2-Fabric interface: Requests missing in L2 are routed through Infinity Fabric to the appropriate memory location, which could be:

Local HBM on the same GPU

Remote GPU memory (in multi-GPU systems)

System memory (CPU DRAM)

The interface categorizes requests by type (read and/or write), size (32B and/or 64B), and destination for optimal routing.

Memory coherence
GPU memory coherence differs significantly from CPU designs to optimize for throughput over latency:

Write-through L1 caches: All writes update both L1 and L2, ensuring L2 always has the latest data. This eliminates the need for complex coherence protocols between L1 caches but requires higher write bandwidth.

Software-managed coherence: Coherence between CUs requires explicit synchronization through:

Memory fences for ordering

Cache invalidation instructions

Atomic operations (executed at L2 level)

Kernel boundaries (implicit synchronization)

Write combining: To handle partial cache line updates from different CUs, the GPU uses write masks indicating which bytes to update. This prevents false sharing issues while maintaining correctness.

Memory coalescing
Memory coalescing combines memory accesses from multiple threads into fewer transactions, significantly improving bandwidth utilization. The coalescing hardware in the VMEM unit analyzes addresses from all threads in a warp and groups them into the minimum number of cache line requests.

Coalesced access pattern: When consecutive threads access consecutive memory addresses, the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data size and alignment).

Non-coalesced access pattern: When threads access widely separated addresses, each thread can generate a separate memory transaction, reducing effective bandwidth by up to 16x or more.

To achieve optimal memory performance:

Ensure consecutive threads access consecutive memory addresses

Align data structures to cache line boundaries (64B and/or 128B)

Use structure-of-arrays rather than array-of-structures layouts

Consider padding to avoid bank conflicts

Architecture variants
AMD supports multiple GPU architecture families optimized for different use cases while maintaining HIP compatibility.

CDNA architecture
CDNA (Compute DNA) specializes in high-performance computing and machine learning workloads. Key features include:

Block diagram showing CDNA3 compute unit with matrix core unit, shader cores, L1 cache, and local data share
CDNA3 compute unit with matrix acceleration
Matrix Core Unit: Specialized hardware for matrix multiply-accumulate operations, providing significantly more throughput than vector units for supported operations. Matrix cores support multiple precisions (INT8, FP16, BF16, FP32) with varying performance characteristics.

Accumulation VGPRs (AGPRs): Additional register file space (up to 256 KB) dedicated to matrix accumulation, doubling the available register storage for matrix operations. Data movement between VGPRs and AGPRs uses specialized instructions (v_accvgpr_*).

Enhanced memory bandwidth: CDNA GPUs typically use HBM2, HBM2e, and/or HBM3 memory technology.

Multi-die designs: CDNA2 (MI250) and CDNA3 (MI300) use chiplet architectures with multiple dies connected through high-speed links, scaling to higher compute and memory capacities.

RDNA architecture
RDNA optimizes for graphics and lower-latency compute workloads through fundamental architectural changes:

Block diagram showing RDNA3 work group processor with dual compute units, shared caches, and 32-wide SIMD units
RDNA3 work group processor architecture
Wave32 execution: Primary execution mode uses 32-thread warps, reducing divergence penalties and register pressure.

Dual compute units: The work group processor (WGP) replaces standalone CUs, containing two closely coupled compute units sharing resources:

Each CU has two 32-wide SIMD units

Warps execute in a single cycle on 32-wide SIMDs

Reduced instruction latency improves responsiveness

Three-level cache hierarchy:

L0 cache: Per-CU cache

L1 cache: Shared between CUs in a WGP (new intermediate level)

L2 cache: Global cache shared across all WGPs

128-byte cache lines: Aligning with Wave32 access patterns (32 threads × 4 bytes = 128 bytes).

These RDNA optimizations target gaming workloads where latency matters more than pure throughput, though the architecture remains capable for general compute tasks.

Performance considerations
Understanding hardware characteristics helps you optimize GPU applications for maximum performance.

Occupancy and resource limits
Occupancy measures the ratio of active warps to maximum possible warps on a CU. Higher occupancy generally improves latency hiding but is limited by:

Register usage: Each warp requires VGPRs and SGPRs from finite pools

LDS allocation: Shared memory used per workgroup

warp slots: Fixed number of execution contexts per CU

Workgroup size: Smaller workgroups can waste resources

Balancing these resources is critical for achieving optimal occupancy. Tools such as rocprofv3 can help analyze occupancy and identify limiting factors.

Latency hiding through multithreading
GPUs hide memory and instruction latency through massive hardware multithreading rather than complex CPU techniques such as out-of-order execution and/or speculation. With sufficient warps:

Memory latency is hidden by executing other warps during waits

Pipeline latencies are covered by round-robin warp scheduling

No context switch overhead as all contexts remain resident

The hardware can switch between warps every cycle, maintaining high ALU utilization even with long-latency operations in flight.

Memory bandwidth utilization
Effective memory bandwidth depends on access patterns:

Coalesced access: Can achieve 70-90% of peak bandwidth

Random access: Might achieve only 5-15% of peak bandwidth

Bank conflicts: Can serialize LDS access, reducing throughput

Memory-bound kernels should focus on:

Maximizing coalescing through proper data layout

Prefetching and data reuse in LDS

Balancing computation with memory access

Using appropriate cache policies

Hardware-specific optimizations
Different AMD GPU architectures benefit from tailored optimizations:

For CDNA:

Optimize for 64-thread warp granularity

Leverage matrix cores for applicable algorithms

Consider AGPR usage for register spilling

For RDNA:

Design for 32-thread warp execution

Utilize improved divergence handling

Take advantage of additional cache level

Architecture-agnostic:

Minimize divergent control flow

Ensure memory access coalescing

Balance resource usage for occupancy

Overlap computation with memory access

Summary
AMD GPU hardware architecture provides massive parallelism through hierarchical organization of compute resources, specialized execution units, and a sophisticated memory system. Understanding these hardware details, from the command processor through shader engines to individual compute units and the memory hierarchy, enables you to write more efficient GPU applications.

Key hardware concepts for optimization include:

Workgroup scheduling and resource management by the SPI

Instruction scheduling and warp execution in compute units

Memory coalescing and cache behavior

Architecture-specific features (matrix cores, Wave32 and/or Wave64 modes)

Resource limits affecting occupancy

For details on mapping parallel algorithms to this hardware, see the Introduction to the HIP programming model chapter. For specific optimization techniques, consult the performance optimization guides in the ROCm documentation.

.. meta::
   :description: How to debug using HIP.
   :keywords: AMD, ROCm, HIP, debugging, ltrace, ROCgdb, WinGDB

.. _debugging_with_hip:

*************************************************************************
Debugging with HIP
*************************************************************************

HIP debugging tools include `ltrace <https://ltrace.org/>`_ and :doc:`ROCgdb <rocgdb:index>`. External tools are available and can be found online. For example, if you're using Windows, you can use Microsoft Visual Studio and WinGDB.

You can trace and debug your code using the following tools and techniques.

Tracing
================================================

You can use tracing to quickly observe the flow of an application before reviewing the detailed
information provided by a command-line debugger. Tracing can be used to identify issues ranging
from accidental API calls to calls made on a critical path.

ltrace is a standard Linux tool that provides a message to ``stderr`` on every dynamic library call. You
can use ltrace to visualize the runtime behavior of the entire ROCm software stack.

Here's a simple command-line example that uses ltrace to trace HIP APIs and output:

.. code-block:: console

    $ ltrace -C -e "hip*" ./hipGetChanDesc
    hipGetChanDesc->hipCreateChannelDesc(0x7ffdc4b66860, 32, 0, 0) = 0x7ffdc4b66860
    hipGetChanDesc->hipMallocArray(0x7ffdc4b66840, 0x7ffdc4b66860, 8, 8) = 0
    hipGetChanDesc->hipGetChannelDesc(0x7ffdc4b66848, 0xa63990, 5, 1) = 0
    hipGetChanDesc->hipFreeArray(0xa63990, 0, 0x7f8c7fe13778, 0x7ffdc4b66848) = 0
    PASSED!
    +++ exited (status 0) +++


Here's another example that uses ltrace to trace hsa APIs and output:

.. code-block:: console

    $ ltrace -C -e "hsa*" ./hipGetChanDesc
    libamdhip64.so.4->hsa_init(0, 0x7fff325a69d0, 0x9c80e0, 0 <unfinished ...>
    libhsa-runtime64.so.1->hsaKmtOpenKFD(0x7fff325a6590, 0x9c38c0, 0, 1) = 0
    libhsa-runtime64.so.1->hsaKmtGetVersion(0x7fff325a6608, 0, 0, 0) = 0
    libhsa-runtime64.so.1->hsaKmtReleaseSystemProperties(3, 0x80084b01, 0, 0) = 0
    libhsa-runtime64.so.1->hsaKmtAcquireSystemProperties(0x7fff325a6610, 0, 0, 1) = 0
    libhsa-runtime64.so.1->hsaKmtGetNodeProperties(0, 0x7fff325a66a0, 0, 0) = 0
    libhsa-runtime64.so.1->hsaKmtGetNodeMemoryProperties(0, 1, 0x9c42b0, 0x936012) = 0
    ...
    <... hsaKmtCreateEvent resumed> )                = 0
    libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6690) = 0
    libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202749000, 4096, 0x7fff325a6690, 0) = 0
    libhsa-runtime64.so.1->hsaKmtCreateEvent(0x7fff325a6700, 0, 0, 0x7fff325a66f0) = 0
    libhsa-runtime64.so.1->hsaKmtAllocMemory(1, 0x100000000, 576, 0x7fff325a67d8) = 0
    libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 8192, 64, 0x7fff325a6790) = 0
    libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273c000, 8192, 0x7fff325a6790, 0) = 0
    libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 4160, 0x7fff325a6450) = 0
    libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273a000, 4096, 0x7fff325a6450, 0) = 0
    libhsa-runtime64.so.1->hsaKmtSetTrapHandler(1, 0x7f120273a000, 4096, 0x7f120273c000) = 0
    <... hsa_init resumed> )                         = 0
    libamdhip64.so.4->hsa_system_get_major_extension_table(513, 1, 24, 0x7f1202597930) = 0
    libamdhip64.so.4->hsa_iterate_agents(0x7f120171f050, 0, 0x7fff325a67f8, 0 <unfinished ...>
    libamdhip64.so.4->hsa_agent_get_info(0x94f110, 17, 0x7fff325a67e8, 0) = 0
    libamdhip64.so.4->hsa_amd_agent_iterate_memory_pools(0x94f110, 0x7f1201722816, 0x7fff325a67f0, 0x7f1201722816 <unfinished ...>
    libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 0, 0x7fff325a6744, 0x7fff325a67f0) = 0
    libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 1, 0x7fff325a6748, 0x7f1200d82df4) = 0
    ...
    <... hsa_amd_agent_iterate_memory_pools resumed> ) = 0
    libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 17, 0x7fff325a67e8, 0) = 0
    <... hsa_iterate_agents resumed> )               = 0
    libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0, 0x7fff325a6850, 3) = 0
    libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0xa000, 0x9e7cd8, 0) = 0
    libamdhip64.so.4->hsa_agent_iterate_isas(0x9dbf30, 0x7f1201720411, 0x7fff325a6760, 0x7f1201720411) = 0
    libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 0, 0x7fff325a6728, 1) = 0
    libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 1, 0x9e7f90, 0) = 0
    libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 4, 0x9e7ce8, 0) = 0
    ...
    <... hsa_amd_memory_pool_allocate resumed> )     = 0
    libamdhip64.so.4->hsa_ext_image_create(0x9dbf30, 0xa1c4c8, 0x7f10f2800000, 3 <unfinished ...>
    libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6740) = 0
    libhsa-runtime64.so.1->hsaKmtQueryPointerInfo(0x7f1202736000, 0x7fff325a65e0, 0, 0) = 0
    libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202736000, 4096, 0x7fff325a66e8, 0) = 0
    <... hsa_ext_image_create resumed> )             = 0
    libamdhip64.so.4->hsa_ext_image_destroy(0x9dbf30, 0x7f1202736000, 0x9dbf30, 0 <unfinished ...>
    libhsa-runtime64.so.1->hsaKmtUnmapMemoryToGPU(0x7f1202736000, 0x7f1202736000, 4096, 0x9c8050) = 0
    libhsa-runtime64.so.1->hsaKmtFreeMemory(0x7f1202736000, 4096, 0, 0) = 0
    <... hsa_ext_image_destroy resumed> )            = 0
    libamdhip64.so.4->hsa_amd_memory_pool_free(0x7f10f2800000, 0x7f10f2800000, 256, 0x9e76f0) = 0
    PASSED!

Debugging
================================================

You can use ROCgdb for debugging and profiling.

ROCgdb is the ROCm source-level debugger for Linux and is based on GNU Project debugger (GDB).
the GNU source-level debugger, equivalent of CUDA-GDB, can be used with debugger frontends, such as Eclipse, Visual Studio Code, or GDB dashboard.
For details, see (https://github.com/ROCm/ROCgdb).

Below is a sample how to use ROCgdb run and debug HIP application, ROCgdb is installed with ROCM package in the folder /opt/rocm/bin.

.. code-block:: console

    $ export PATH=$PATH:/opt/rocm/bin
    $ rocgdb ./hipTexObjPitch
    GNU gdb (rocm-dkms-no-npi-hipclang-6549) 10.1
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    ...
    For bug reporting instructions, please see:
    <https://github.com/ROCm/ROCgdb/issues>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    ...
    Reading symbols from ./hipTexObjPitch...
    (gdb) break main
    Breakpoint 1 at 0x4013d1: file /home/test/hip/tests/src/texture/hipTexObjPitch.cpp, line 98.
    (gdb) run
    Starting program: /home/test/hip/build/directed_tests/texture/hipTexObjPitch
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

    Breakpoint 1, main ()
        at /home/test/hip/tests/src/texture/hipTexObjPitch.cpp:98
    98	    texture2Dtest<float>();
    (gdb)c

Debugging HIP applications
--------------------------------------------------------------------------------------------

The following Linux example shows how to get useful information from the debugger while running a
simple memory copy test, which caused a segmentation fault issue.

.. code-block:: console

    test: simpleTest2<?> numElements=4194304 sizeElements=4194304 bytes
    Segmentation fault (core dumped)

    (gdb) run
    Starting program: /home/test/hipamd/build/directed_tests/runtimeApi/memory/hipMemcpy_simple
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

    Breakpoint 1, main (argc=1, argv=0x7fffffffdea8)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:147
    147     int main(int argc, char* argv[]) {
    (gdb) c
    Continuing.
    [New Thread 0x7ffff64c4700 (LWP 146066)]

    Thread 1 "hipMemcpy_simpl" received signal SIGSEGV, Segmentation fault.
    0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    104             A_h1[i] = 3.14f + 1000 * i;
    (gdb) bt
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    #1  0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163
    (gdb) info thread
    Id   Target Id                                            Frame
    * 1    Thread 0x7ffff64c5880 (LWP 146060) "hipMemcpy_simpl" 0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    2    Thread 0x7ffff64c4700 (LWP 146066) "hipMemcpy_simpl" 0x00007ffff6b0850b in ioctl
        () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) thread 2
    [Switching to thread 2 (Thread 0x7ffff64c4700 (LWP 146066))]
    #0  0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
    #1  0x00007ffff6604568 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #2  0x00007ffff65fe73a in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #3  0x00007ffff659e4d6 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #4  0x00007ffff65807de in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #5  0x00007ffff65932a2 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #6  0x00007ffff654f547 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #7  0x00007ffff7f76609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
    #8  0x00007ffff6b13293 in clone () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) thread 1
    [Switching to thread 1 (Thread 0x7ffff64c5880 (LWP 146060))]
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    104             A_h1[i] = 3.14f + 1000 * i;
    (gdb) bt
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    #1  0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163
    (gdb)
    ...

Debugging HIP applications using Windows tools can be more informative than on Linux. Windows
tools provides more visibility into debug codes, which makes it easier to inspect variables, watch
multiple details, and examine call stacks.

HIP Record & Replay
===================

HIP Record & Replay (HRR) captures HIP API traces into a binary archive and replays them on a live GPU for bug reproduction, debugging, and validation. During capture, HRR records every HIP API call made by an application into a binary archive (.hrr directory). During replay, it reproduces the original workload on a live GPU, including multi-threaded command submission, graph execution, and GPU memory transfers. 

As described in https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hrr/README.md
HRR allows you to capture and replay runtime operations of a HIP kernel, letting you replay the
kernel on other GPUs without having the avilable source code. This lets you review performance issues
or execution problems without the actual application. 

Useful environment variables
===================================================

HIP provides environment variables that allow HIP, hip-clang, or HSA drivers to prevent certain features
and optimizations. These are not intended for production, but can be useful to diagnose
synchronization problems in the application (or driver).

Some of the more widely used environment variables are described in this section. These are
supported on the Linux ROCm path and Windows.

Kernel enqueue serialization
---------------------------------------------------------------------------------

You can control kernel command serialization from the host:

``AMD_SERIALIZE_KERNEL``, for serializing kernel enqueue
 ``AMD_SERIALIZE_KERNEL = 1``, Wait for completion before enqueue
 ``AMD_SERIALIZE_KERNEL = 2``, Wait for completion after enqueue
 ``AMD_SERIALIZE_KERNEL = 3``, Both

Or

``AMD_SERIALIZE_COPY``, for serializing copies
 ``AMD_SERIALIZE_COPY = 1``, Wait for completion before enqueue
 ``AMD_SERIALIZE_COPY = 2``, Wait for completion after enqueue
 ``AMD_SERIALIZE_COPY = 3``, Both

So HIP runtime can wait for GPU idle before/after any GPU command depending on the environment
setting.

Making device visible
---------------------------------------------------------------------------------

For systems with multiple devices, you can choose to make only certain device(s) visible to HIP using
``HIP_VISIBLE_DEVICES``. Once enabled, HIP can only view devices that have indices present in the sequence.
For example:

.. code-block:: console

    $ HIP_VISIBLE_DEVICES=0,1

Or in the application:

.. code-block:: cpp

    if (totalDeviceNum > 2) {
    setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
    assert(getDeviceNumber(false) == 3);
    ... ...
    }

Dump code object
---------------------------------------------------------------------------------

To analyze compiler-related issues, you can use the dump code object:
``GPU_DUMP_CODE_OBJECT``.

HSA-related environment variables (Linux)
-----------------------------------------------------------------------------------------------

HSA provides environment variables that help analyze issues in drivers or hardware.

* To isolate issues with hardware copy engines, you can use ``HSA_ENABLE_SDMA``.

  ``HSA_ENABLE_SDMA=0`` causes host-to-device and device-to-host copies to use compute shader
  blit kernels, rather than the dedicated DMA copy engines. Compute shader copies have low latency
  (typically < 5 us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.

* To diagnose interrupt storm issues in the driver, you can use ``HSA_ENABLE_INTERRUPT``.

  ``HSA_ENABLE_INTERRUPT=0`` causes completion signals to be detected with memory-based
  polling, rather than interrupts.

HIP environment variable summary
--------------------------------

Here are some of the more commonly used environment variables:

.. include-table:: ./reference/env_variables/debug_hip_env.rst
    :table: hip-env-debug

General debugging tips
======================================================

* ``gdb --args`` can be used to pass the executable and arguments to ``gdb``.

* You can set environment variables (``set env``) from within GDB on Linux:

  .. code-block:: bash

      (gdb) set env AMD_SERIALIZE_KERNEL 3

  .. note::

      This ``gdb`` command does not use an equal (=) sign.

* The GDB backtrace shows a path in the runtime. This is because a fault is caught by the runtime, but it is generated by an asynchronous command running on the GPU.

* To determine the true location of a fault, you can force the kernels to run synchronously by setting the environment variables ``AMD_SERIALIZE_KERNEL=3`` and ``AMD_SERIALIZE_COPY=3``. This forces HIP runtime to wait for the kernel to finish running before returning. If the fault occurs when a kernel is running, you can see the code that launched the kernel inside the backtrace. The thread that's causing the issue is typically the one inside ``libhsa-runtime64.so``.

* VM faults inside kernels can be caused by:

  * Incorrect code (e.g., a for loop that extends past array boundaries)

  * Memory issues, such as invalid kernel arguments (null pointers, unregistered host pointers, bad pointers)

  * Synchronization issues

  * Compiler issues (incorrect code generation from the compiler)

  * Runtime issues
  Understanding GPU performance
This topic explains the theoretical foundations of GPU performance on AMD hardware. Understanding these concepts helps you analyze performance characteristics, identify bottlenecks, and make informed optimization decisions.

For practical optimization techniques and step-by-step guidance, see Performance guidelines.

Performance bottlenecks
The neck of a bottle limits the rate at which liquid can be poured. A performance bottleneck in a computing system similarly limits the rate at which work can be completed.

A performance bottleneck is the limiting factor that prevents a GPU kernel from achieving higher performance. Understanding which bottleneck applies helps identify the appropriate optimization approach.

Performance bottlenecks for GPU kernels fall into three main categories:

Compute-bound: The kernel is limited by arithmetic throughput (the arithmetic bandwidth of compute units)

Memory-bound: The kernel is limited by memory bandwidth (how quickly data can move between High Bandwidth Memory (HBM) and on-chip caches or Local Data Share (LDS))

Overhead-bound: The kernel is limited by latency (host-side scheduling, kernel launch overhead, or small array operations)

This categorization aligns with the textbook approach to optimization: determine the bottleneck, elevate the bottleneck until it is no longer limiting, and repeat on the new bottleneck.

Roofline model analysis helps quickly identify whether a kernel’s performance is bottlenecked by compute throughput or memory bandwidth.

Roofline model
The roofline model is a simplified, visual model of performance used to quickly determine whether a program is limited by memory bandwidth or arithmetic bandwidth.

In the roofline model, two hardware-derived “roofs” place upper bounds—or ceilings—on achievable performance:

Compute roof: The peak arithmetic rate of the target hardware (vector Arithmetic Logic Units (ALUs) or Matrix Fused Multiply-Add (MFMA) units), sometimes referred to as its arithmetic bandwidth

Memory roof: The peak data transfer rate of the memory subsystem, or memory bandwidth

These are plotted on a plane with arithmetic intensity (operations per byte) on the x-axis and performance (operations per second) on the y-axis.

The compute roof is a horizontal line at a height equal to the hardware’s maximum arithmetic throughput. The memory roof is a slanted line whose slope equals the memory bandwidth (in bytes per second). Because slope is “rise over run,” its units correspond to throughput per intensity.

Roofline model diagram showing memory bandwidth ceiling and compute ceiling
Roofline model showing the relationship between arithmetic intensity and achievable performance. The memory bandwidth ceiling represents the GPU’s memory bandwidth limit, while the compute ceiling shows the maximum achievable TFLOPs. Kernels falling into the area to the left of the ridge point are memory-bound, while they are compute-bound if they fall into the right area.
A kernel’s position on the x-axis indicates whether it is fundamentally compute-bound (beneath the flat roof) or memory-bound (beneath the slanted roof). In practice, few kernels ever fully reach either roof due to overhead, latency, and control-divergence effects.

The point where the diagonal and horizontal roofs intersect is called the ridge point. Its x-coordinate gives the minimum arithmetic intensity required to escape the memory bottleneck. Systems with ridge points farther to the left are easier to saturate, but over time, improvements in compute throughput have far outpaced memory growth—pushing ridge points steadily to the right.

On AMD platforms, roofline analysis is built into ROCm’s profiling and performance visualization ecosystem. rocprofv3 can be used to gather achieved FLOPs, memory transactions, and operational intensity.

The roofline model’s elegance lies in its simplicity—but also in its deliberate omissions. It ignores latency entirely, focusing only on sustained throughput limits. Understanding those assumptions—and when they hold—is essential to applying the model correctly.

Compute-bound performance
A kernel is compute-bound when its performance is limited by the GPU’s arithmetic throughput rather than memory bandwidth. These kernels have high arithmetic intensity, spending most cycles executing arithmetic operations.

Kernels that are compute-bound are limited by the arithmetic bandwidth of the GPU’s Compute Units (CUs)—on AMD architectures, this means the vector ALUs and MFMA units within each CU or Single Instruction Multiple Data (SIMD) unit.

Characteristics of compute-bound kernels:

High ratio of arithmetic operations to memory accesses (high arithmetic intensity)

Performance scales with GPU compute capacity

Limited benefit from memory bandwidth optimization

Can often achieve a high percentage of peak theoretical FLOPS

The limiting factor is utilization of arithmetic pipelines: the number of concurrent floating-point or integer operations the GPU can sustain per clock

The theoretical maximum is determined by:

Number of compute units and SIMD lanes

Clock frequency

Instruction throughput per cycle

Specialized unit capabilities (matrix cores, Special Function Units (SFUs))

Memory-bound performance
A kernel is memory-bound when its performance is limited by memory bandwidth rather than compute capacity. These kernels have low arithmetic intensity and spend significant time waiting for memory operations.

Kernels that are memory-bound are limited by the memory bandwidth of the GPU—that is, by how quickly data can move between HBM and the on-chip caches or LDS of the GPU’s compute units.

Memory-bound kernels are limited by the bandwidth between GPU RAM and local caches because the working sets of most real-world GPU workloads are far larger than any higher level of the memory hierarchy. When data reuse is low and arithmetic operations per byte are few, the speed of computation is dominated by how quickly memory can feed operands to the arithmetic units.

Characteristics of memory-bound kernels:

Low ratio of arithmetic operations to memory accesses (low arithmetic intensity)

Performance scales with memory bandwidth

Sensitive to memory access patterns

Typically achieve lower percentage of peak FLOPS

Fall to the left of the ridge point on the roofline diagram

The theoretical maximum is determined by:

HBM bandwidth capacity

Memory controller efficiency

Cache hierarchy effectiveness

Memory access pattern efficiency

Arithmetic intensity
Arithmetic intensity is the ratio of arithmetic operations to memory operations in a kernel. It is the ratio of floating-point operations (FLOPs) to memory traffic (bytes) for a given kernel or algorithm.

 
This metric determines whether a kernel is compute-bound or memory-bound.

Key points:

Higher arithmetic intensity indicates more computation per byte transferred

A high arithmetic intensity indicates that a kernel performs many arithmetic operations per byte loaded

The balance point (ridge point) depends on the GPU’s compute-to-bandwidth ratio

It can be calculated theoretically or measured empirically

Different precision types affect both FLOPs and bytes

For modern AMD GPUs:

The compute-to-bandwidth ratio varies by GPU generation

Higher-end models have higher ratios

Kernels above the GPU’s specific ratio (ridge point) are compute-bound

Because modern GPUs deliver far more arithmetic throughput than memory bandwidth, the most efficient kernels are those with high arithmetic intensity

Algorithmic complexity and intensity scaling
Because algorithms have different operational and memory complexities, they scale differently in arithmetic intensity:

An algorithm with 
 operations and 
 memory has 
 
 intensity (decreasing with size)

One with 
 operations and 
 memory has 
 intensity (increasing with size)

Examples of kernel complexity scaling:

SAXPY (
): 
, 
 → intensity 
 
 → 
 scaling

Single-Precision Real Fast Fourier Transform (FFT): 
 
, 
 → intensity 
 
 → 
 scaling

SGEMM (matrix multiplication): 
, 
 → 
 scaling

Matrix multiplication scales linearly in arithmetic intensity— 
 operations versus 
 memory—making it an ideal match for high-throughput architectures. This favorable scaling is a key reason why many machine-learning algorithms built around dense linear algebra achieve high GPU utilization.

Techniques that shift memory transfers to additional compute operations reduce memory traffic but increase arithmetic load, thereby raising the arithmetic intensity. For example:

Compressing data in global memory reduces bytes transferred but adds decompression arithmetic—raising arithmetic intensity

In training and inference of neural networks, techniques like gradient checkpointing reduce memory storage of activations (fewer bytes stored and loaded) but add recompute work—again increasing arithmetic intensity

Latency hiding mechanisms
GPUs hide memory and instruction latency through massive hardware multithreading rather than complex CPU techniques like out-of-order execution.

Latency hiding is the strategy of masking long-latency operations by running them concurrently. On AMD GPUs, performant kernels interleave the execution of many threads across warps keeping overall throughput high even when individual instructions take many cycles. When one warp stalls on a slow global-memory access, the scheduler immediately issues instructions from another eligible warp.

How latency hiding works:

warp switching: Context switches occur every cycle with zero overhead

Multiple warps per CU: Many concurrent warps supported

Instruction-level parallelism: Multiple independent instructions in flight

This keeps the compute units busy: while one warp drives MFMA matrix ops, another runs scalar and vector ALU work (e.g., quantize and dequantize), and a third issues loads and stores through the memory pipeline (LDS, L1, and L2 ↔ HBM).

The hardware can completely hide memory latency if there are enough active warps with independent work. The number of instructions required from other warps to hide latency depends on the GPU’s specific memory latency and instruction throughput characteristics.

Little’s Law
Little’s Law relates concurrency (how much work is in flight) to latency and throughput:

In GPU terms, it tells you how much independent work you must have in flight to hide latency via fine-grained scheduling. On AMD GPUs, warp switching by the warp schedulers is the primary latency-hiding mechanism.

Little’s Law determines how many independent memory transactions or instructions must be outstanding across active warps to keep the compute units busy.

Concretely, consider a simple sequence in AMD CDNA terms:

global_load_dword    v1, v[2:3], off   ; long-latency global load (hundreds of cycles)
v_mul_lo_u32         v2, v1, 0xBEEF    ; integer multiply
v_add_u32            v4, v2, 0xAFFE    ; integer add
v_mul_lo_u32         v6, v4, 0x1337    ; integer multiply
Executed strictly serially, the total time is dominated by the global load. Using Little’s Law, if your effective issue rate is ~1 inst/cycle and the load takes hundreds of cycles, you need hundreds of independent in-flight operations to finish, on average, one such sequence per cycle—hiding the memory latency from consumers.

Issuance occurs at the warp granularity (64 threads per warp on CDNA). When latency hiding is successful, the CU maintains enough active warps and rapidly context-switches among them whenever one blocks, so execution units don’t sit idle waiting on memory or other long-latency events.

Requirements for effective latency hiding:

Sufficient occupancy (active warps)

Independent instructions to overlap

Balanced resource usage

Minimal divergence

Warp (Wavefront) execution states
The state of the warps executing a kernel on an AMD GPU can be described using several non-exclusive terms—active, stalled, eligible, and selected.

A warp is considered active from the time its threads begin executing until all threads in that warp have completed the kernel. The warp schedulers select warps from the active pool each cycle; the selected warps then issue their instructions.

An eligible warp is an active warp ready to issue its next instruction. For a warp to be eligible:

Its next instruction has been fetched

The required pipeline (vector ALU, MFMA, or memory) is available

All data dependencies have been resolved

No synchronization barriers (for example, s_barrier) are pending

Eligible warps are the immediate candidates for issue. A lack of eligible warps often indicates dependency or memory stalls—a key target in performance tuning.

A stalled warp is active but unable to issue its next instruction due to resource or data hazards. Common causes include:

Execution dependencies: waiting for results from previous ALU or MFMA operations

Memory dependencies: waiting for global or LDS memory fetches

Pipeline conflicts: required execution units are occupied

AMD hardware uses a scoreboard mechanism to track outstanding dependencies per warp. When waiting on LDS or ALU results, a warp is said to be on the short scoreboard; when waiting on off-chip HBM accesses, it is on the long scoreboard. This scoreboarding approach—originally from the CDC 6600 supercomputer—allows dynamic scheduling across warps (thread-level parallelism) rather than within them (instruction-level parallelism).

A selected warp is an eligible one chosen by the warp scheduler to issue an instruction in the current cycle. Each CU typically has multiple schedulers that can each issue one instruction per cycle from their eligible pool.

Understanding these states helps explain GPU utilization metrics:

Active cycles: Percentage of cycles with at least one instruction executing

Stall cycles: Percentage of cycles waiting for resources

Idle cycles: No warps available to execute

Maximizing active cycles while minimizing stall and idle cycles improves performance. Effective latency hiding on AMD hardware relies on keeping enough active and eligible warps resident so that the schedulers always have work to select, ensuring the CU pipelines remain fully utilized.

Occupancy theory
Occupancy measures the ratio of active warp to the maximum possible warps on a compute unit.

 
There are two common ways to measure it:

Theoretical occupancy: The upper limit determined by the kernel’s launch configuration (threads per block, register use, LDS use) and the hardware limits of the CU

Achieved occupancy: The actual number of warps active during kernel execution, i.e., on active cycles

As part of the AMD execution model, all threads in a block are scheduled to the same CU. Each CU has finite resources—Vector General-Purpose Registers (VGPRs), Scalar General-Purpose Registers (SGPRs), LDS (shared memory), and wave slots—that must be shared among all resident blocks. These constraints jointly determine the maximum number of active warps.

Why occupancy matters:

Higher occupancy improves latency hiding

More concurrent warps mask memory and instruction latency

Enables better utilization of execution units

Limiting factors:

Register usage: VGPRs and SGPRs per thread

Shared memory (LDS): Allocation per block

Warp slots: Hardware limit on concurrent warps

Block size: Small blocks may waste resources

Trade-offs:

Higher occupancy improves latency hiding but reduces resources per thread

Lower occupancy allows more resources per thread but may expose latency

Optimal occupancy depends on kernel characteristics

Memory-bound kernels benefit more from high occupancy

Low occupancy often reduces performance when there aren’t enough eligible warps to hide memory or arithmetic latency, causing low issue efficiency and underutilized pipelines. However, once occupancy is sufficient for latency hiding, increasing it further can hurt performance by reducing the number of available registers or LDS per warp—both of which can limit arithmetic intensity.

In short, occupancy measures how fully a CU is loaded, not how efficiently it is utilized. High-performance kernels (for example, MFMA-based GEMMs on CDNA) often operate at low occupancy because only a few warps are needed to fully saturate the MFMA and memory pipelines.

Memory hierarchy impact on performance
The GPU memory hierarchy has different bandwidths and latencies:

Memory types by speed:

Registers: Fastest, lowest latency (per-thread storage)

LDS (shared memory): Very fast, on-chip (per-block storage)

L1 cache: Fast, on-chip (per-CU cache)

L2 cache: Moderate, on-chip (shared across CUs)

HBM (global memory): Slower, off-chip but high bandwidth

Memory coalescing theory
Memory coalescing is a hardware technique that improves effective memory bandwidth by servicing many logical loads or stores with a small number of physical memory transactions.

Memory coalescing combines memory accesses from multiple threads into fewer transactions. When consecutive threads access consecutive memory addresses, the hardware can merge requests into efficient cache line accesses.

Memory coalescing is relevant when accessing global memory (HBM and GDDR attached to the GPU). For efficient access to LDS and shared memory, see the discussion of bank conflicts below.

On AMD GPUs, global memory is backed by HBM (in data-center parts) or GDDR (on many client parts). These Dynamic Random-Access Memory (DRAM) technologies provide very high bandwidth but have relatively long access latency. If each thread’s load or store were always turned into its own physical DRAM transaction, the GPU would leave a large fraction of that raw bandwidth unused.

Coalescing takes advantage of how DRAM is organized internally. When a DRAM address is accessed, the hardware actually fetches or writes a burst: a run of consecutive addresses fetched together in a single transaction. If multiple threads in a warp access addresses that fall into the same burst, those logical accesses can be coalesced into that single transaction.

This fits naturally with the warp execution model: in normal execution, all threads in a warp execute the same instruction at the same time. If each lane in the warp loads or stores a value from a contiguous region (e.g., lane i accesses base + i), the memory system can typically serve the entire warp’s request with a small number of large, aligned bursts. When addresses are scattered (e.g., large strides or irregular indexing), more bursts are needed, and effective bandwidth drops.

Why coalescing matters:

Reduces number of memory transactions

Improves memory bandwidth utilization

Decreases memory access latency

Coalesced pattern: Consecutive threads accessing consecutive addresses achieve high bandwidth utilization.

Non-coalesced pattern: Random or strided addresses result in many separate transactions and low bandwidth utilization.

Example: strided access pattern
Consider this kernel that reads from an input array with a configurable stride (distance between consecutive elements accessed by each thread):

__global__ void strided_read_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    std::size_t N, std::size_t stride)
{
    const std::size_t t  = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t T  = gridDim.x * blockDim.x;

    float acc = 0.f;

    for (std::size_t j = t * stride; j < N; j += T * stride)
    {
        // across a warp, addresses differ by (stride * sizeof(float))
        float v = in[j]; // perfectly coalesced for stride == 1
        acc = acc * 1.000000119f + v;  // force compiler to keep the load
    }

    // one write per thread (negligible vs reads)
    if (t < N) out[t] = acc;
}
When stride == 1, threads in the same warp access consecutive 4-byte elements (in[j], in[j+1], in[j+2], …). These accesses fall into a small number of aligned DRAM bursts, so the memory system can coalesce them efficiently and deliver high bandwidth.

As stride increases:

The addresses accessed by neighboring threads move farther apart

Each warp’s loads spread across more bursts

The number of physical transactions per logical access increases

Effective bandwidth drops

Bank conflict theory
Shared memory (LDS) is organized into banks that can be accessed independently. Bank conflicts occur when multiple threads access different addresses in the same bank.

A bank conflict occurs when multiple threads in a warp simultaneously access different addresses that reside in the same LDS bank. When that happens, the accesses to that bank must be serialized, reducing effective LDS throughput by an integer factor and preventing full utilization of the on-chip memory bandwidth.

On AMD GPUs, the on-chip LDS (HIP’s “shared memory”) inside each compute unit is physically organized into banks. These banks can be accessed in parallel, which is how LDS achieves very high bandwidth. On CDNA and RDNA architectures, LDS is divided into 32 (CDNA, CDNA2 and CDNA3) or 64 (CDNA4+ and RDNA2+) banks, each 4 bytes wide. Conceptually, addresses map to banks like this (low bits only, in bytes):

Address: 0x00 0x04 0x08 0x0C 0x10 0x14 0x18 0x1C ... 0x7C
Bank:       0    1    2    3    4    5    6    7 ...   31

Address: 0x80 0x84 0x88 0x8C 0x90 0x94 0x98 0x9C ... 0xFC
Bank:       0    1    2    3    4    5    6    7 ...   31
Any two addresses that differ by 
 map to the same bank.

Why bank conflicts matter:

Conflicts serialize accesses, reducing throughput

LDS bandwidth drops proportionally to conflict degree

Can turn parallel operations into sequential ones

Common patterns:

No conflict: Each thread accesses a different bank (full bandwidth)

Broadcast: Multiple threads read the same address (no conflict)

N-way conflict: N threads access the same bank (1/N bandwidth)

Example: conflict-free access
If you access sequential elements of a float array in LDS, different lanes in a warp naturally land in different banks:

__shared__ float data[1024];  // in HIP, __shared__ maps to LDS

int tid = threadIdx.x;
float value = data[tid];  // addresses: 0x00, 0x04, 0x08, ...
For 32 consecutive float elements, this maps cleanly: each 4-byte word goes to a different bank (0–31). All these accesses can be serviced in one LDS transaction, so there are no bank conflicts. This is the “good” pattern.

Example: pathological strided access (conflict-heavy)
Now consider a pattern that walks down a column of a row-major LDS array where each row has 32 floats:

__shared__ float data[32 * 32];  // 32 columns per row

int tid = threadIdx.x;
float value = data[tid * 32];    // addresses: 0x00, 0x80, 0x100, ...
// recall: sizeof(float) == 4 bytes
Here, each successive access is offset by 
 —exactly one full bank span. So:

data[0] → bank 0

data[32] → bank 0

data[64] → bank 0

Every lane in the warp is hitting bank 0, but at different addresses, all in the same cycle. These accesses must be serialized by the LDS hardware, which can turn a fast LDS access into a slow operation.

When conflicts don’t happen
If all threads in a warp access the exact same address in a bank (e.g., all lanes reading the same control value from LDS), hardware can often broadcast that value. In that case, the request is not treated as a conflict—it’s one read, fanned out to many lanes.

Register pressure theory
Register pressure refers to a situation where the register file becomes a performance bottleneck due to excessive demand for registers by active threads.

Register pressure occurs when a kernel requires more registers than optimal for the target occupancy.

In GPU programming, registers are the fastest level of the memory hierarchy, holding per-thread variables for arithmetic and address computation. However, while the compiler (amdclang++ for ROCm) works with an unbounded set of virtual registers, the hardware has only a finite number of physical registers per compute unit.

How register pressure arises
Each thread in a warp consumes a number of registers as determined by the compiled Instruction Set Architecture (ISA) code (AMD CDNA or RDNA assembly). All threads in a work-group share the same CU, so the total register file usage per work-group depends on both:

The number of threads per work-group (launch configuration), and

The number of registers required per thread (kernel complexity)

As the register footprint per work-group increases, fewer work-groups can be resident on a CU at once. This directly reduces occupancy, which in turn limits the ability of the GPU to hide latency through thread-level parallelism. In extreme cases, the compiler may even “spill” register values into global memory—orders of magnitude slower than register access.

Why register pressure matters:

Reduces maximum occupancy

May cause register spilling to memory

Decreases ability to hide latency

Lowers overall throughput

The relationship between registers and occupancy:

More registers per thread → fewer concurrent warps

Fewer registers per thread → higher occupancy but may need memory spills

Optimal balance depends on kernel memory access patterns

Performance metrics explained
Understanding performance metrics is essential for analyzing GPU behavior:

Peak rate
Peak rate is the theoretical maximum throughput a hardware system can achieve. It represents the absolute upper bound of GPU performance—the architecture’s effective ‘speed of light.’

Peak rate assumes ideal conditions: every compute unit is fully active, all execution pipelines are perfectly fed, and no constraints (e.g., register pressure, memory stalls, synchronization, or bandwidth limits) impede progress.

The theoretical maximum performance of a GPU:

Peak FLOPS: Maximum floating-point operations per second

Peak bandwidth: Maximum memory throughput

Peak instruction rate: Maximum instructions per cycle

In performance analysis, peak rate serves several roles:

It defines the compute-bound “roof” in a roofline model

It forms the denominator for utilization metrics such as pipe utilization or CU utilization

It provides the theoretical yardstick against which achieved performance (measured FLOPs per second) is compared

Actual performance is always below peak due to various inefficiencies.

Utilization metrics
Pipe utilization
Pipe utilization measures how effectively a kernel uses the execution pipelines within each compute unit.

Each CU contains multiple independent execution pipes, each specialized for a different class of operations—for example:

VALU pipes handle general vector arithmetic (add, multiply, Fused Multiply-Add (FMA))

MFMA pipes handle matrix operations on matrix-fused multiply–accumulate units

SALU pipes handle scalar operations

Load and Store (LDS and VMEM) units handle memory access instructions

Branch and Control units handle program flow

Pipe utilization quantifies the percentage of each pipeline’s peak theoretical throughput that is being achieved, averaged over all active CUs and cycles in which the pipeline is active.

Pipe utilization: The percentage of execution cycles where the pipeline is actively processing instructions. Low utilization indicates stalls or insufficient work.

Before analyzing performance at the level of pipe utilization, you should first examine kernel utilization (how often CUs are busy) and CU utilization (how evenly work is distributed across CUs). Once those are sufficient, per-pipe metrics reveal whether the performance limit is arithmetic, memory, or control-bound.

On AMD GPUs, these measurements are exposed through ROCm profiling tools such as rocprofv3.

Relevant counters include:

SQ_ACCUM_PREV_HI_BUSY (VALU pipeline busy percentage)

SQ_ACCUM_MFMA_BUSY (MFMA utilization)

SQ_ACCUM_LDS_BUSY and SQ_ACCUM_VMEM_BUSY (memory pipelines)

SQ_ACCUM_SALU_BUSY (scalar ALU activity)

Together, these form the pipe utilization profile—showing how well each instruction pipeline is being fed with eligible warps and how close the kernel is to saturating the hardware’s arithmetic or memory throughput.

Issue efficiency
Issue efficiency measures how effectively the warp scheduler on each compute unit keeps the execution pipelines busy by issuing instructions from eligible warps. In a perfectly efficient kernel, the scheduler issues one instruction every cycle for every active CU.

An issue efficiency of 100% means that on every active cycle, at least one warp was eligible and an instruction was successfully issued. Lower values indicate that during some cycles, all active warps were stalled—waiting on memory, dependencies, or resources—and the scheduler was idle, reducing total instruction throughput.

Issue efficiency: The ratio of issued instructions to the maximum possible. Low efficiency can indicate instruction cache misses, scheduling inefficiencies, or resource conflicts.

On AMD GPUs, issue efficiency can be measured using hardware performance counters exposed through ROCProfiler or Omnitrace, such as:

SQ_WAVES_BUSY — percentage of cycles where any warp was actively executing

SQ_WAVES_ISSUED — number of issued waves per cycle

SQ_ACCUM_INSTS_ISSUED — total instructions issued per CU

SQ_ACCUM_CYCLES_BUSY — number of cycles the CU was active

By combining these metrics, you can estimate how efficiently the scheduler keeps the CU’s pipelines fed. Low issue efficiency typically signals insufficient concurrency (low occupancy) or high memory latency, both of which prevent the hardware from issuing instructions continuously.

CU utilization
CU utilization measures the percentage of time that compute units on an AMD GPU are actively executing instructions.

Instead of reporting the fraction of time a kernel is executing somewhere on the GPU, CU utilization reports the fraction of time all CUs spend executing warps.

CU utilization: The percentage of compute units actively executing work. Low utilization suggests insufficient parallelism, load imbalance, or synchronization overhead.

As with GPU utilization, high CU utilization is generally desirable. It indicates that most CUs are busy executing instructions across the device. However, high CU utilization alone does not guarantee full performance.

If CU utilization is high but throughput remains low, the kernel may not be effectively using the functional pipelines within each CU—such as vector ALUs, MFMA tensor cores, or load and store units. In that case, you should examine pipe utilization, which measures how fully those individual execution paths are being used.

CU utilization can be observed with AMD’s profiling and monitoring tools:

amd-smi metric --usage — reports overall GPU activity percentage

rocprofv3 — provides performance counters like SQ_WAVE_CYCLES, SQ_BUSY_CYCLES, and per-pipe instruction metrics

rocminfo — shows the number of CUs available per GPU

In summary, CU utilization captures how actively the GPU’s compute units are engaged in running warps. High CU utilization indicates good parallel workload distribution, while low CU utilization may point to poor occupancy, launch configuration limits, or insufficient concurrency.

Branch efficiency
Branch efficiency measures how often all threads within a warp take the same execution path when encountering conditional statements.

It quantifies control-flow uniformity—that is, how often all lanes in a warp evaluate a conditional identically. It is calculated as the ratio of uniform branch decisions to total branch instructions executed. High branch efficiency indicates little to no warp divergence, while low branch efficiency means many lanes are masked off due to diverging control flow.

Branch efficiency: The ratio of non-divergent to total branches. Low efficiency indicates significant divergence overhead.

Not all conditionals reduce branch efficiency. The common “bounds check” pattern found in most GPU kernels, for instance:

int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n)
usually has very high branch efficiency, since nearly all warps consist entirely of threads that either satisfy idx < n or not—except perhaps for the last partial warp, which straddles the boundary of n.

While CPUs also optimize branch behavior, they focus on temporal uniformity—predicting whether the same branch will be taken or not over repeated iterations. GPUs, on the other hand, care about spatial uniformity: whether all lanes in the warp take the same branch at the same time.

On AMD architectures, this spatial uniformity is tracked via the EXEC mask. Divergence forces EXEC to toggle individual bits to deactivate lanes following a different control path. High branch efficiency implies minimal EXEC manipulation, meaning nearly all lanes execute the same instruction stream simultaneously—maximizing SIMD efficiency and overall throughput.

Theoretical performance limits
Understanding theoretical limits helps set realistic performance expectations.

Peak performance bounds
Every GPU has theoretical maximum performance determined by:

Clock frequency and number of compute units

Instruction throughput per clock cycle

Memory bandwidth capacity

Specialized unit capabilities (matrix cores, SFUs)

Achievable performance
Real applications typically achieve a fraction of theoretical peak due to:

Imperfect resource utilization

Memory access inefficiencies

Control flow divergence

Synchronization overhead

Launch and scheduling costs

The gap between theoretical and achieved performance reveals optimization opportunities. The roofline model provides a framework for understanding these limits and identifying which factor (compute or memory) constrains performance.

Summary
Understanding GPU performance requires knowledge of several interconnected concepts:

Performance bottlenecks: Whether compute, memory, or overhead limits performance

Roofline model: Visual framework for analyzing performance limits based on arithmetic intensity

Arithmetic intensity: The compute-to-memory ratio of algorithms

Latency hiding: How concurrent execution masks delays through warp switching

Occupancy: How warp concurrency affects resource utilization

Memory hierarchy: How different memory types affect bandwidth and the importance of coalescing

Performance metrics: Quantitative measures for analysis including pipe utilization, issue efficiency, CU utilization, and branch efficiency

These theoretical foundations inform practical optimization decisions. For step-by-step optimization techniques and practical guidance, see Performance guidelines.


.. meta::
  :description: This chapter describes a set of best practices designed to help
   developers optimize the performance of HIP-capable GPU architectures.
  :keywords: AMD, ROCm, HIP, CUDA, performance, guidelines, optimization, how-to

.. _how_to_performance_guidelines:

*******************************************************************************
Performance guidelines
*******************************************************************************

The AMD HIP performance guidelines provide practical, actionable techniques for
optimizing application performance on AMD GPUs. This guide focuses on
step-by-step instructions and best practices for improving performance.

For theoretical foundations and performance concepts, see
:doc:`../understand/performance_optimization`.

Optimization workflow
=====================

Follow this systematic approach to optimize GPU performance:

1. **Profile and measure baseline**

   Use ``rocprofv3`` to identify bottlenecks:

   .. code-block:: bash

      rocprofv3 --stats --<tracing_option> -- <application_path>

   Collect metrics on kernel execution time, memory bandwidth, occupancy, and
   CU utilization. For more details on using ``rocprofv3`` for application
   tracing and profiling, see :doc:`rocprofv3 documentation
   <rocprofiler-sdk:how-to/using-rocprofv3>`.

2. **Analyze metrics to identify bottlenecks**

   Determine if kernels are compute-bound or memory-bound. Check arithmetic
   intensity, memory bandwidth achieved vs peak, and compute throughput.

   For understanding the roofline model, see :ref:`roofline_model`.

3. **Apply targeted optimizations**

   Based on identified bottlenecks, apply techniques from this guide.

4. **Verify improvements**

   Re-profile to confirm performance gains.

5. **Iterate**

   Repeat until performance goals are met.

Profiling and analysis tools
=============================

ROCm provides a comprehensive suite of profiling and analysis tools that help developers understand and optimize GPU performance. These tools are essential for identifying bottlenecks and evaluating the effectiveness of performance optimizations.

rocprofv3
---------

The tool :doc:`rocprofv3 <rocprofiler-sdk:how-to/using-rocprofv3>` provides
command-line-driven profiling for detailed performance analysis. It collects
metrics on kernel execution time, memory bandwidth, warp occupancy, VALU
utilization, and instruction-level counters.

``rocprofv3`` integrates with the
:doc:`rocProfiler-SDK framework <rocprofiler-sdk:index>` to collect hardware
traces and API-level timing data. The collected data can be exported in JSON and
CSV formats for further analysis or visualization.

Key capabilities:

* Kernel execution profiling
* Memory bandwidth analysis
* Warp occupancy metrics
* Compute unit utilization
* Instruction-level performance counters
* API call tracing
* Hardware event collection

Trace visualization with Perfetto
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For application-level tracing, ``rocprofv3`` can generate traces compatible with
`Perfetto <https://ui.perfetto.dev>`__, a third-party, open-source trace viewer.
This enables visualization of the complete timeline of application execution,
including the temporal relationships among host operations, kernel launches,
memory transfers, and synchronization events.

Using Perfetto with traces generated by ``rocprofv3`` can help identify
performance issues caused by API overhead, inefficient synchronization, or
insufficient overlap between computation and data movement.

The following trace data is available for visualization:

* Application execution timelines
* HIP API call tracking and timing
* ROCm library call analysis
* Host-device synchronization events
* Memory transfer operations

ROCprof Compute Viewer
----------------------

`ROCprof Compute Viewer <https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/latest/>`__
provides a GUI-based environment for analyzing GPU kernel performance data. It
delivers detailed kernel-level insights, including counter correlation and
hierarchical performance breakdowns, helping developers interpret execution
patterns and identify optimization opportunities.

Key features:

* Kernel performance counter analysis
* Counter correlation and visualization
* Hierarchical kernel performance breakdown
* Interactive performance data exploration
* Kernel-level optimization insights


AMD System Management Interface
--------------------------------

The :doc:`AMD System Management Interface <amdsmi:index>` (AMD SMI) is a
command-line utility for querying, monitoring, and managing AMD GPUs. It
provides system administrators and developers with detailed, real-time
information about GPU hardware, utilization, and power metrics.

``amd-smi`` reports the following categories of information:

* GPU identity information, such as the card name, device ID, and PCI bus
  location
* Live utilization metrics, including GPU activity, memory usage, clock speeds,
  and active processes
* Power and thermal readings, such as temperature, fan speed, voltage, and
  power draw
* Performance states and limits, such as available frequency levels, clock
  throttling, and voltage controls

These metrics are retrieved through the AMD SMI C API, which exposes a stable,
scriptable interface for system and performance monitoring tools.

``amd-smi`` also supports management operations, including:

* Setting power caps and performance profiles
* Adjusting clock frequencies
* Performing GPU resets and controlling persistence mode
* Reporting or controlling ECC status on supported data center GPUs

Output can be formatted as human-readable text or JSON (``--json``), commonly
used for integration into automated monitoring pipelines.

Basic usage examples:

.. code-block:: shell

   # Display GPU information
   amd-smi static

   # Monitor GPU utilization and metrics
   amd-smi metric --usage

   # Show detailed information in JSON format
   amd-smi static --json

Typical workflow
----------------

For comprehensive ROCm performance analysis:

1. Use ``rocprofv3`` to collect profiling data and generate traces
2. Use ROCprof Compute Viewer for detailed kernel performance analysis and
   counter correlation
3. Use third-party tools like Perfetto to visualize application traces, API
   calls, and timeline behavior
4. Use ``amd-smi`` to monitor GPU utilization and system health
5. Iterate between profiling and optimization, verifying improvements with
   each change

This multi-tool approach provides kernel-level metrics, application-level
tracing, system monitoring, and visual context to understand overall
performance behavior.

.. _parallel execution:

Parallel execution
==================

For optimal use and to keep all system components busy, the application must
reveal and efficiently provide as much parallelism as possible.

Application level
-----------------

To enable parallel execution across the host and devices:

* Use :ref:`asynchronous calls and streams <asynchronous_how-to>`
* Assign serial workloads to the host
* Assign parallel workloads to the devices

For parallel workloads:

* Use :cpp:func:`__syncthreads()` (see :ref:`synchronization_functions`) for
  intra-block synchronization
* Use global memory with separate kernel invocations for inter-block
  synchronization (has overhead, minimize when possible)

Device level
------------

Maximize parallel execution across multiprocessors:

* Execute multiple kernels concurrently on a device
* Use streams to overlap computation and data transfers
* Keep all multiprocessors busy with enough concurrent kernels
* Avoid launching too many kernels (causes resource contention)

Multiprocessor level
--------------------

Maximize parallel execution within each :ref:`compute unit <compute_unit>`:

* Ensure sufficient resident :ref:`warps <wavefront>` for every clock cycle
* Exploit instruction-level parallelism within warps
* Exploit thread-level parallelism across warps
* Balance resource usage for optimal :ref:`occupancy <occupancy>`

.. _memory optimization:

Memory throughput optimization
==============================

The first step in maximizing memory throughput is to minimize low-bandwidth
data transfers between the host and the device.

Additionally, maximize the use of on-chip memory (shared memory and caches) and
minimize transfers with global memory.

.. _data transfer:

Data transfer optimization
--------------------------

**Minimize host-device transfers**

* Move computations from host to device when possible
* Create, use, and discard intermediate data structures on device
* Avoid unnecessary copies to host memory

**Batch small transfers**

Each memory transfer incurs a fixed overhead from driver calls and PCIe
transaction setup. Consolidating many small transfers into a single large
transfer amortizes this overhead across more data, resulting in much higher
effective bandwidth.

.. code-block:: cuda

   // Instead of many small transfers
   for (int i = 0; i < n; i++) {
       hipMemcpy(&d_data[i], &h_data[i], sizeof(float), ...);
   }

   // Use a single large transfer
   hipMemcpy(d_data, h_data, n * sizeof(float), ...);

**Use page-locked memory for transfers**

Page-locked (pinned) memory cannot be swapped to disk by the operating system,
allowing the GPU to access it directly via DMA without CPU involvement. This
eliminates an extra copy through a staging buffer and achieves higher bandwidth.

.. code-block:: cuda

   float* h_pinned;
   hipHostMalloc(&h_pinned, size);
   // Faster transfers than pageable memory
   hipMemcpy(d_data, h_pinned, size, hipMemcpyHostToDevice);

**Use mapped memory on integrated systems**

On integrated GPUs (APUs), the CPU and GPU share the same physical memory.
Mapped page-locked memory allows zero-copy access, where the GPU reads directly
from host memory without requiring an explicit transfer, eliminating redundant
copies.

.. code-block:: cuda

   int integrated;
   hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device);
   if (integrated) {
       // Use mapped page-locked memory - no explicit copy needed
       hipHostMalloc(&ptr, size, hipHostMallocMapped);
   }

.. _device memory access:

Device memory access
--------------------

**Ensure proper alignment**

Memory hardware loads data in aligned chunks (typically 128 bytes). Using
naturally aligned data types ensures each access maps to a single memory
transaction, maximizing bandwidth and avoiding split transactions.

.. code-block:: cuda

   // Use naturally aligned types
   float4 data;  // 16-byte aligned
   float2 data;  // 8-byte aligned

   // Ensure structure alignment
   struct __align__(16) MyStruct {
       float4 data;
   };

**Optimize 2D array access**

Padding 2D arrays to multiples of the warp size ensures each row starts at an
aligned memory boundary. This allows consecutive threads accessing the same row
to generate coalesced memory transactions, thereby maximizing bandwidth.

.. code-block:: cuda

   // Ensure array width is multiple of warp size
   int width = ((actual_width + warpSize - 1) / warpSize) * warpSize;
   hipMalloc(&array, width * height * sizeof(float));

   // Access pattern
   int idx = x + width * y;  // width should be warp-aligned

**Coalesce memory accesses**

When consecutive threads in a warp access consecutive memory addresses, the
hardware combines these into a single wide transaction. Non-coalesced patterns
require multiple transactions, reducing effective bandwidth.

.. code-block:: cuda

   // Good: consecutive threads access consecutive addresses
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   data[idx] = value;

   // Bad: strided access
   int idx = threadIdx.x * stride;  // Non-coalesced if stride > 1
   data[idx] = value;

For understanding memory coalescing theory, see :ref:`memory_hierarchy_theory`.

**Use shared memory for data reuse**

Shared memory (:ref:`LDS <lds>`) provides fast on-CU scratchpad memory for
communication between threads in a block. Loading data into shared memory once
and reusing it many times reduces global memory traffic, particularly effective
for tiled algorithms such as matrix multiplication.

.. code-block:: cuda

   __global__ void optimized_kernel(float* input, float* output) {
       __shared__ float tile[TILE_SIZE][TILE_SIZE];

       // Load data into shared memory
       tile[threadIdx.y][threadIdx.x] = input[...];
       __syncthreads();

       // Reuse data from fast shared memory
       float result = 0;
       for (int i = 0; i < TILE_SIZE; i++) {
           result += tile[threadIdx.y][i] * tile[i][threadIdx.x];
       }
       __syncthreads();

       output[...] = result;
   }

**Avoid bank conflicts in shared memory**

Shared memory is organized into banks, each capable of servicing one request
per cycle. When multiple threads in a :ref:`warp <wavefront>` access the
same bank simultaneously, the requests are serialized, reducing throughput.
Padding arrays by one element shifts addresses to avoid systematic conflicts.

.. code-block:: cuda

   // Bad: power-of-2 stride causes conflicts
   __shared__ float data[32][32];
   float value = data[threadIdx.x][threadIdx.y];

   // Good: padding avoids conflicts
   __shared__ float data[32][33];  // Extra column
   float value = data[threadIdx.x][threadIdx.y];

For bank conflict theory, see :ref:`bank_conflicts_theory`.

**Use texture memory for 2D spatial access**

Texture memory provides hardware-accelerated 2D filtering and caching optimized
for spatial locality. It automatically handles boundary conditions and can
interpolate values, making it ideal for image processing and nearby-neighbor
access patterns.

.. code-block:: cuda

   // Create texture object
   hipTextureObject_t texObj;
   hipCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

   // Access in kernel
   float value = tex2D<float>(texObj, x, y);

.. _instruction optimization:

Instruction throughput optimization
====================================

Arithmetic instructions
-----------------------

**Use efficient operations**

Division requires many more hardware cycles than multiplication. Similarly,
bitwise operations (shifts, AND, OR) are single-cycle instructions on integer
units, making them far more efficient than equivalent arithmetic for
power-of-two calculations.

.. code-block:: cuda

   // Prefer multiplication over division
   float result = value * 0.5f;     // Fast
   float result = value / 2.0f;     // Slower

   // Use bitwise operations for powers of 2
   int index = threadIdx.x << 2;    // Multiply by 4
   int mask = (1 << n) - 1;         // Create bit mask

**Use single-precision when possible**

AMD GPUs have significantly higher throughput for single-precision (FP32)
operations compared to double-precision (FP64). Using single-precision math
functions can deliver substantial performance gains when FP64 accuracy is not
required.

.. code-block:: cuda

   // Single-precision (faster)
   float result = sinf(x);
   float result = expf(x);

   // Double-precision (slower, use only when necessary)
   double result = sin(x);
   double result = exp(x);

**Leverage fast math intrinsics**

Hardware-specific intrinsics bypass certain accuracy checks and use lookup
tables or polynomial approximations, trading slight precision loss for
significantly higher throughput. These should be used when the application can
tolerate reduced precision.

.. code-block:: cuda

   // Fast intrinsic versions
   float ex = __expf(x);            // Fast exponential
   float lg = __logf(x);            // Fast logarithm
   float sq = __fsqrt_rn(x);        // Fast square root
   float rc = __frcp_rn(x);         // Fast reciprocal

.. _control flow instructions:

Control flow optimization
-------------------------

**Minimize divergence**

When threads in a warp take different execution paths, the hardware serializes
both branches, executing each path with only the relevant threads active. This
reduces effective parallelism and wastes cycles on inactive threads.

.. code-block:: cuda

   // Good: no divergence (condition depends on threadIdx)
   if (threadIdx.x < 32) {
       // All threads in first half-warp execute
   }

   // Bad: divergence within warp
   if (data[threadIdx.x] > threshold) {
       // Some threads execute, others don't
   }

**Use branch hints for predictable conditions**

Providing hints about branch likelihood helps the compiler generate better
instruction ordering and can improve the branch predictor's accuracy, reducing
pipeline stalls when the prediction proves correct.

.. code-block:: cuda

   if (__builtin_expect(rare_condition, 0)) {
       // Unlikely branch
   }

   // C++20 attribute
   if (common_condition) [[likely]] {
       // Likely branch
   }

**Avoid divergent warps**

When divergence is unavoidable, restructure the code to separate divergent paths
into different kernel launches or use predication (branchless programming) to
keep all threads active, though computing unnecessary values may be acceptable
if it avoids the serialization penalty.

.. code-block:: cuda

   // Instead of:
   if (threadIdx.x % 2 == 0) {
       result = compute_even();
   } else {
       result = compute_odd();
   }

   // Consider separating into different kernels or using predication

Synchronization
---------------

**Use minimal synchronization**

Each synchronization point stalls all threads in a block until the slowest one
reaches the barrier. Minimize synchronizations by carefully analyzing data
dependencies—only synchronize when threads genuinely need to exchange data
through shared memory.

.. code-block:: cuda

   __global__ void kernel() {
       __shared__ float data[256];

       // Load phase
       data[threadIdx.x] = input[...];
       __syncthreads();  // Necessary sync

       // Compute phase - no sync needed if threads are independent
       float result = compute(data[...]);

       // Store phase - sync only if needed
       output[...] = result;
   }

**Use streams for async execution**

Streams enable concurrent execution of independent operations. Commands in
different streams can overlap in time, allowing kernel execution and memory
transfers to run simultaneously. This maximizes GPU utilization by keeping
multiple execution engines busy concurrently.

.. code-block:: cuda

   hipStream_t stream1, stream2;
   hipStreamCreate(&stream1);
   hipStreamCreate(&stream2);

   // Overlap independent operations
   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block, 0, stream2>>>(...);

   hipStreamSynchronize(stream1);
   hipStreamSynchronize(stream2);

Managing register pressure
==========================

High register usage can limit :ref:`occupancy <occupancy>`. Follow these steps:

**Minimize live variables**

The compiler allocates registers for every variable that must remain accessible.
Reducing the number of simultaneously live variables frees registers, allowing
more warps to fit on each CU. Chaining function calls trades some redundant
computation for lower register usage.

.. code-block:: cuda

   // Instead of storing all intermediate results
   float a = compute_a();
   float b = compute_b();
   float c = compute_c();
   float result = combine(a, b, c);

   // Recompute or chain operations
   float result = combine(compute_a(), compute_b(), compute_c());

**Use shared memory for temporary storage**

Per-thread arrays stored in registers consume valuable register space, limiting
:ref:`occupancy <occupancy>`. Moving temporary storage to
:ref:`shared memory <lds>` trades register usage for shared memory usage, often
allowing higher occupancy since shared memory limits are typically less
restrictive.

.. code-block:: cuda

   // Instead of per-thread arrays (uses registers)
   float temp[100];

   // Use shared memory
   __shared__ float temp[blockDim.x][100];
   float* my_temp = temp[threadIdx.x];

**Adjust launch bounds**

The ``__launch_bounds__`` attribute provides hints to the compiler about
expected thread block size and minimum blocks per CU. This guides register
allocation decisions, potentially trading per-thread register count for higher
occupancy.

.. code-block:: cuda

   __global__ void
   __launch_bounds__(256, 4)  // 256 threads, 4 blocks per CU
   my_kernel() {
       // Kernel code
   }

**Check register usage during compilation**

The compiler can report per-kernel register usage statistics. Monitoring this
output helps identify kernels consuming excessive registers, guiding
optimization efforts toward reducing register pressure in the most impactful
areas.

.. code-block:: shell

   hipcc --resource-usage kernel.hip

For register pressure theory, see :ref:`register_pressure_theory`.

Improving occupancy
===================

Higher :ref:`occupancy <occupancy>` helps hide latency. Follow these steps:

**Reduce register usage per thread**

Use techniques from "Managing register pressure" above.

**Reduce shared memory usage per block**

Each :ref:`CU <compute_unit>` has limited :ref:`shared memory <lds>` that must
be divided among resident blocks. Reducing per-block shared memory usage allows
more blocks to reside simultaneously, increasing :ref:`occupancy <occupancy>`
and improving latency hiding through greater thread-level parallelism.

.. code-block:: cuda

   // Allocate only what's needed
   __shared__ float tile[TILE_SIZE][TILE_SIZE];

   // Or use dynamic allocation
   extern __shared__ float dynamic_shared[];

**Optimize block size**

AMD Instinct GPUs execute threads in :ref:`warps <wavefront>` of 64, while
AMD Radeon GPUs execute threads in warps of 32. Choosing
block sizes as multiples of 64 or 32 prevents partial warps that waste
execution slots. Larger blocks (128-256 threads) typically achieve better
:ref:`occupancy <occupancy>` and resource utilization.

.. code-block:: cuda

   // Use multiples of warp size
   dim3 block(64);    // Good for AMD Instinct GPUs (warp=64)
   dim3 block(128);   // Common choice
   dim3 block(256);   // Good for high-occupancy kernels

   // Avoid very small blocks
   dim3 block(32);    // May waste resources on Instinct GPUs

**Profile occupancy**

Profiling tools report the ratio of active :ref:`warps <wavefront>` to
maximum possible warps per :ref:`CU <compute_unit>`. Low
:ref:`occupancy <occupancy>` suggests resource constraints (registers or shared
memory) are limiting parallelism and may indicate opportunities for
optimization.

.. code-block:: shell

   rocprofv3 --occupancy ./your_application

For occupancy theory, see :ref:`occupancy`.

Minimizing memory thrashing
============================

Applications frequently allocating and freeing memory might experience slower
allocation calls over time. To optimize:

**Allocate early, deallocate late**

Frequent allocation and deallocation causes memory fragmentation and increases
allocator overhead. Reusing allocations across iterations amortizes the cost
of memory management and maintains better memory locality.

.. code-block:: cuda

   // Bad: frequent allocation in loop
   for (int i = 0; i < iterations; i++) {
       float* temp;
       hipMalloc(&temp, size);
       // Use temp
       hipFree(temp);
   }

   // Good: allocate once
   float* temp;
   hipMalloc(&temp, size);
   for (int i = 0; i < iterations; i++) {
       // Reuse temp
   }
   hipFree(temp);

**Avoid allocating all available memory**

Reserving some memory headroom prevents allocation failures and system
instability. The driver and runtime need workspace for internal operations, and
leaving a safety margin ensures stable operation without unexpected
out-of-memory errors.

.. code-block:: cuda

   std::size_t free, total;
   hipMemGetInfo(&free, &total);

   // Don't allocate all free memory
   std::size_t safe_size = free * 0.9;  // Leave some margin

**Use managed memory for oversubscription**

Managed memory automatically migrates data between host and device on demand,
allowing allocations larger than physical GPU memory. Prefetching hints help
the runtime optimize page placement, reducing migration overhead during kernel
execution.

.. code-block:: cuda

   // Allows exceeding physical memory
   float* data;
   hipMallocManaged(&data, large_size);

   // Optionally prefetch to device
   hipMemPrefetchAsync(data, size, device, stream);

Summary
=======

Key optimization techniques:

* **Profile first**: Use ``rocprofv3`` to identify actual bottlenecks
* **Parallelize effectively**: Maximize work at all levels (application, device,
  CU)
* **Optimize memory**: Minimize transfers, maximize coalescing, use LDS
* **Manage resources**: Balance registers, shared memory, and occupancy
* **Minimize divergence**: Structure control flow to keep
  :ref:`warps <wavefront>` coherent

For understanding the theory behind these techniques, refer to
:doc:`../understand/performance_optimization` and
:doc:`../understand/hardware_implementation`.
