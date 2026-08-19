// Focused unit test #3: GPU EFT (integer-split TwoProduct, no fma) vs trusted
// CPU oracle that sets the SSE (MXCSR) rounding mode and does the single-op.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <hip/hip_runtime.h>

static inline __host__ __device__ uint64_t u64(double x) { return *reinterpret_cast<uint64_t*>(&x); }
static inline __host__ __device__ double d64(uint64_t x) { return *reinterpret_cast<double*>(&x); }

__host__ __device__ double rx_next_after_up(double x) {
	if (x == 0.0) return 0x1.0p-1074;
	if (x == INFINITY) return INFINITY;
	if (x == -INFINITY) return -0x1.fffffffffffffp1023;
	uint64_t u = u64(x);
	if (u & 0x8000000000000000ULL) u--; else u++;
	return d64(u);
}
__host__ __device__ double rx_next_after_down(double x) {
	if (x == 0.0) return -0x1.0p-1074;
	if (x == -INFINITY) return -INFINITY;
	if (x == INFINITY) return 0x1.fffffffffffffp1023;
	uint64_t u = u64(x);
	if (u & 0x8000000000000000ULL) u++; else u--;
	return d64(u);
}
__host__ __device__ double2 rx_two_sum(double a, double b) {
	double hi = a + b; double v = hi - a; double lo = (a - (hi - v)) + (b - v); return make_double2(hi, lo);
}
// Integer-bit Veltkamp split (no overflow, no fma).
__host__ __device__ double rx_split(double x) {
	uint64_t bits = u64(x);
	uint64_t sign = bits & 0x8000000000000000ULL;
	int e = (int)((bits >> 52) & 0x7ff);
	uint64_t m = bits & 0x000FFFFFFFFFFFFFULL;
	if (e == 0) {
		if (m == 0) return x;
		// Subnormal x = M*2^-1074 (M = m, 52-bit mantissa). Split M into high 27
		// and low 25 bits; restore the subnormal scale so ah+al == x exactly.
		uint64_t H = m >> 25;                 // high 27 bits
		uint64_t L = m & 0x1FFFFFFULL;        // low 25 bits
		(void)L;
		return d64(sign | (H << 25));         // ah = H*2^25 * 2^-1074 ; al = x - ah recovers L*2^-1074
	}
	if (e == 0x7ff) return x;
	uint64_t F = m | 0x0010000000000000ULL;
	uint64_t top27 = F >> 26;
	uint64_t drop = F & 0x3FFFFFFULL;
	if (drop > 0x2000000ULL) top27 += 1;
	else if (drop == 0x2000000ULL && (top27 & 1)) top27 += 1;
	if (top27 == 0x8000000ULL) { top27 = 0; e += 1; }
	else top27 = (top27 & 0x3FFFFFFULL) << 26;
	if (e >= 0x7ff) return d64(sign | (0x7ffULL << 52));
	return d64(sign | (((uint64_t)e) << 52) | top27);
}
__host__ __device__ double2 rx_two_prod(double a, double b) {
	double hi = a * b;
	if (hi == 0.0 || !isfinite(hi)) return make_double2(hi, 0.0);
	double ah = rx_split(a); double al = a - ah;
	double bh = rx_split(b); double bl = b - bh;
	double lo = ((ah * bh - hi) + ah * bl + al * bh) + al * bl;
	return make_double2(hi, lo);
}
__host__ __device__ double2 rx_dd_add(double2 x, double2 y) {
	double2 s = rx_two_sum(x.x, y.x); double lo = s.y + x.y + y.y; return rx_two_sum(s.x, lo);
}
__host__ __device__ double2 rx_dd_sub(double2 x, double2 y) {
	return rx_dd_add(x, make_double2(-y.x, -y.y));
}
// ulp(s) = distance from s to the next representable double (in the +inf direction).
__host__ __device__ double rx_ulp(double s) {
	if (!isfinite(s)) return 0.0;
	if (s == 0.0) return 0x1.0p-1074;
	return rx_next_after_up(s) - s;
}
// Compare a double-double (hi,lo) to a scalar t.
__host__ __device__ int rx_cmp_dd(double hi, double lo, double t) {
	if (hi > t) return 1;
	if (hi < t) return -1;
	if (lo > 0.0) return 1;
	if (lo < 0.0) return -1;
	return 0;
}
// Tie to even between two consecutive doubles a < b.
__host__ __device__ double rx_tie_even(double a, double b) {
	return (u64(a) & 1) ? b : a;
}
// Given s = some near-rounded value and the EXACT residual r = v - s (double-double,
// |r| <= thr), return the correctly-rounded value of v under 'mode' (0=RN,1=RD,2=RU,3=RZ).
// 'thr' is the residual magnitude corresponding to 1 ulp(s); half = 0.5*thr.
__host__ __device__ double rx_directed_round_thr(double s, double rhi, double rlo, int mode, double thr) {
	if (!isfinite(s)) return s;
	double half = 0.5 * thr;
	int c0 = rx_cmp_dd(rhi, rlo, 0.0);
	if (c0 == 0) return s;                       // v == s exactly
	int cUp = rx_cmp_dd(rhi, rlo, half);        // r vs +half
	int cDn = rx_cmp_dd(rhi, rlo, -half);       // r vs -half
	int cUlpUp = rx_cmp_dd(rhi, rlo, thr);      // r vs +ulp
	int cUlpDn = rx_cmp_dd(rhi, rlo, -thr);     // r vs -ulp
	double sup = rx_next_after_up(s);
	double sdn = rx_next_after_down(s);
	if (c0 > 0) { // v in (s, sup], maybe == sup
		if (cUlpUp >= 0) return sup;            // |r| >= ulp -> v == sup (or beyond)
		bool aboveHalf = (cUp > 0);
		bool atHalf = (cUp == 0);
		if (mode == 2) return sup;                       // RU
		if (mode == 3) return (s >= 0.0) ? s : sup;     // RZ
		if (mode == 1) return s;                         // RD
		// RN
		if (aboveHalf) return sup;
		if (atHalf) return rx_tie_even(s, sup);
		return s;
	} else { // v in [sdn, s), maybe == sdn
		if (cUlpDn <= 0) return sdn;            // |r| >= ulp
		bool aboveHalf = (cDn < 0);
		bool atHalf = (cDn == 0);
		if (mode == 1) return sdn;                       // RD
		if (mode == 3) return (s <= 0.0) ? s : sdn;     // RZ
		if (mode == 2) return s;                         // RU
		// RN
		if (aboveHalf) return sdn;
		if (atHalf) return rx_tie_even(sdn, s);
		return s;
	}
}
__host__ __device__ double rx_directed_round(double s, double rhi, double rlo, int mode) {
	return rx_directed_round_thr(s, rhi, rlo, mode, rx_ulp(s));
}
__host__ __device__ double rx_underflow_result(int mode, int signv, int iszero) {
	if (iszero) return 0.0;
	if (mode == 0 || mode == 3) return 0.0;
	if (mode == 1) return (signv > 0) ? 0.0 : -0x1.0p-1074;   // toward -inf
	return (signv > 0) ? 0x1.0p-1074 : 0.0;                    // toward +inf (mode 2)
}
__host__ __device__ double rx_fma_rnd(double a, double b, double c, int mode) {
	if (mode == 0) return a * b + c;
	double s0 = a * b + c; if (!isfinite(s0)) return s0;
	double2 v = rx_two_prod(a, b); v = rx_dd_add(v, make_double2(c, 0.0));
	double s = v.x + v.y; double2 r = rx_dd_sub(v, make_double2(s, 0.0));
	if (s == 0.0) {
		int iszero, signv;
		if (c == 0.0) { iszero = (a == 0.0 || b == 0.0); signv = (signbit(a) == signbit(b)) ? 1 : -1; }
		else { iszero = (u64(a) == (u64(c) ^ 0x8000000000000000ULL)); double av = fabs(a), cv = fabs(c); if (av > cv) signv = (signbit(a)) ? -1 : 1; else if (cv > av) signv = (signbit(c)) ? -1 : 1; else signv = 0; }
		return rx_underflow_result(mode, signv, iszero);
	}
	return rx_directed_round(s, r.x, r.y, mode);
}
__host__ __device__ double rx_ddiv(double a, double b, int mode) {
	double q = a / b;
	if (isnan(q)) return q;
	// Overflow: exact |a/b| exceeds max finite. SSE honors rounding mode.
	if (q == INFINITY || q == -INFINITY) {
		double maxf = 0x1.fffffffffffffp1023;
		double half_ulp_maxf = 0x1.0p970;   // 0.5 * ulp(maxf)
		double A = fabs(a), B = fabs(b);
		// M = maxf + 0.5*ulp(maxf); compare A with B*M (exact).
		double2 bmax = rx_two_prod(B, maxf);
		double2 bhalf = rx_two_prod(B, half_ulp_maxf);
		double2 Mprod = rx_dd_add(bmax, bhalf);
		int c = rx_cmp_dd(Mprod.x, Mprod.y, A);   // A vs B*M
		bool over = (c > 0);                       // A > B*M -> |X| > M
		if (q == INFINITY) {                       // X > +maxf
			if (mode == 2) return INFINITY;
			if (mode == 1 || mode == 3) return maxf;
			return over ? INFINITY : (c < 0 ? maxf : INFINITY);   // RN: tie -> +inf (even)
		} else {                                   // X < -maxf
			if (mode == 1) return -INFINITY;
			if (mode == 2 || mode == 3) return -maxf;
			return over ? -INFINITY : (c < 0 ? -maxf : -INFINITY); // RN: tie -> -inf (even)
		}
	}
	if (q == 0.0) { int iszero = (a == 0.0); int signv = (signbit(a) == signbit(b)) ? 1 : -1; return rx_underflow_result(mode, signv, iszero); }
	// two_prod is exact only for normal operands; subnormal operands make its error term
	// collapse to 0. So normalize both a and b to the normal range, do the product on
	// normal operands, then assemble the residual and threshold in the a-normalized domain.
	int Ea, Eb; (void)frexp(a, &Ea); (void)frexp(b, &Eb);
	double a_s = ldexp(a, -Ea);                 // M_a in [0.5,1)  (normal)
	double b_s = ldexp(b, -Eb);                 // M_b in [0.5,1)  (normal)
	double2 qb_s = rx_two_prod(q, b_s);         // exact: q*b_s = qb_s.x + qb_s.y  (normal*normal)
	double k = (double)(Eb - Ea);
	double qbx_s = ldexp(qb_s.x, k);            // q*b scaled by 2^(-Ea)
	double qby_s = ldexp(qb_s.y, k);
	double2 S_s = rx_dd_sub(make_double2(a_s, 0.0), rx_dd_add(make_double2(qbx_s, 0.0), make_double2(qby_s, 0.0)));
	// Threshold (scaled 1-ulp residual) = b*ulp(q)*2^(-Ea) = b_s*ulp(q)*2^(Eb-Ea).
	double thr = ldexp(b_s * rx_ulp(q), k);
	return rx_directed_round_thr(q, S_s.x, S_s.y, mode, thr);
}
__host__ __device__ double rx_dsqrt(double a, int mode) {
	if (!(a >= 0.0)) return sqrt(a);          // negative or NaN -> NaN
	double y = sqrt(a); if (!isfinite(y)) return y;
	if (y == 0.0) { int iszero = (a == 0.0); return rx_underflow_result(mode, 1, iszero); }
	// Scale a by 4^k (k integer) so a' = a*4^k is normal; then sqrt(a') = sqrt(a)*2^k,
	// and the residual r' = (a' - y'^2)/(2 y') satisfies r'/ulp(y') = r/ulp(y).
	// Scaling back by 2^k (power of 2) preserves the directed rounding.
	int Ea; (void)frexp(a, &Ea);
	int twok = ((Ea % 2) == 0) ? -Ea : (1 - Ea);   // Ea+twok in {0,1}, twok even
	int k = twok / 2;
	double ap = ldexp(a, twok);
	double yp = sqrt(ap);
	double2 yyp = rx_two_prod(yp, yp); double2 Sp = rx_dd_sub(make_double2(ap, 0.0), yyp);
	double rp = (Sp.x + Sp.y) / (2.0 * yp);
	double yp_r = rx_directed_round(yp, rp, 0.0, mode);
	return ldexp(yp_r, -k);
}

// Trusted CPU oracle using __float128 exact arithmetic.
// Trusted CPU oracle: SSE scalar ops honor MXCSR rounding (what RandomX reference uses).
static inline unsigned rq_rm(int mode) {
	return (mode == 0) ? 0x0000 : (mode == 1) ? 0x2000 : (mode == 2) ? 0x4000 : 0x6000;
}
static double ref_op(double a, double b, double c, int mode) {
	unsigned orig = _mm_getcsr();
	_mm_setcsr((orig & ~0x6000) | rq_rm(mode));
	__m128d va = _mm_set_sd(a), vb = _mm_set_sd(b), vc = _mm_set_sd(c);
	__m128d r = (mode == 0) ? _mm_add_sd(_mm_mul_sd(va, vb), vc)
		: (c == 0.0) ? _mm_mul_sd(va, vb) : _mm_add_sd(va, vc);
	_mm_setcsr(orig);
	return _mm_cvtsd_f64(r);
}
static double ref_div(double a, double b, int mode) {
	unsigned orig = _mm_getcsr();
	_mm_setcsr((orig & ~0x6000) | rq_rm(mode));
	__m128d r = _mm_div_sd(_mm_set_sd(a), _mm_set_sd(b));
	_mm_setcsr(orig);
	return _mm_cvtsd_f64(r);
}
static double ref_sqrt(double a, int mode) {
	unsigned orig = _mm_getcsr();
	_mm_setcsr((orig & ~0x6000) | rq_rm(mode));
	__m128d r = _mm_sqrt_sd(_mm_set_sd(a), _mm_set_sd(a));
	_mm_setcsr(orig);
	return _mm_cvtsd_f64(r);
}

__global__ void k_fma(double* A, double* B, double* C, int* M, double* O, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) O[i] = rx_fma_rnd(A[i], B[i], C[i], M[i]);
}
__global__ void k_div(double* A, double* B, int* M, double* O, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) O[i] = rx_ddiv(A[i], B[i], M[i]);
}
__global__ void k_sqrt(double* A, int* M, double* O, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) O[i] = rx_dsqrt(A[i], M[i]);
}

int main() {
	const int N = 500000;
	double* hA = new double[N]; double* hB = new double[N]; double* hC = new double[N]; int* hM = new int[N];
	for (int i = 0; i < N; i++) {
		// mix: normals (some large), subnormals, tiny
		uint64_t xa, xb, xc;
		int pick = rand() % 4;
		if (pick == 0) { xa = ((uint64_t)rand()<<32)|(((uint64_t)rand()<<1)); xb = ((uint64_t)rand()<<32)|(((uint64_t)rand()<<1)); } // full range
		else if (pick == 1) { xa = 0x3fe0000000000000ULL + (rand()&0xffffff); xb = 0x3fe0000000000000ULL + (rand()&0xffffff); } // ~1.0
		else if (pick == 2) { xa = 0x7fe0000000000000ULL + (rand()&0x0fffff); xb = 0x0010000000000000ULL + (rand()&0xffffff); } // huge * tiny
		else { xa = 0x0000000000000001ULL + (rand()&0xfffff); xb = 0x0000000000000001ULL + (rand()&0xfffff); } // subnormal
		xc = ((uint64_t)rand()<<32)|(((uint64_t)rand()<<1));
		// avoid inf/nan inputs (skip if not finite)
		if (!isfinite(d64(xa)) || !isfinite(d64(xb)) || !isfinite(d64(xc))) { i--; continue; }
		hA[i] = d64(xa); hB[i] = d64(xb); hC[i] = d64(xc);
		hM[i] = i % 4;
	}
	double* cMul = new double[N]; double* cAdd = new double[N]; double* cD = new double[N]; double* cS = new double[N];
	for (int i = 0; i < N; i++) cMul[i] = ref_op(hA[i], hB[i], 0.0, hM[i]);            // MUL: a*b
	for (int i = 0; i < N; i++) cAdd[i] = ref_op(hA[i], 1.0, hC[i], hM[i]);            // ADD: a+c
	for (int i = 0; i < N; i++) cD[i] = ref_div(hA[i], hB[i], hM[i]);
	for (int i = 0; i < N; i++) cS[i] = ref_sqrt(hA[i], hM[i]);

	double *dA,*dB,*dC,*dF,*dD,*dS; int* dM;
	hipMalloc(&dA,N*8); hipMalloc(&dB,N*8); hipMalloc(&dC,N*8); hipMalloc(&dM,N*4); hipMalloc(&dF,N*8); hipMalloc(&dD,N*8); hipMalloc(&dS,N*8);
	hipMemcpy(dA,hA,N*8,hipMemcpyHostToDevice); hipMemcpy(dB,hB,N*8,hipMemcpyHostToDevice); hipMemcpy(dC,hC,N*8,hipMemcpyHostToDevice); hipMemcpy(dM,hM,N*4,hipMemcpyHostToDevice);
	// separate zero/one arrays so GPU tests exactly the MUL (c=0) and ADD (b=1) paths used by RandomX
	double* hZero = new double[N](); double* hOne = new double[N]; for(int i=0;i<N;i++) hOne[i]=1.0;
	double* dZero; double* dOne; hipMalloc(&dZero,N*8); hipMalloc(&dOne,N*8);
	hipMemcpy(dZero,hZero,N*8,hipMemcpyHostToDevice); hipMemcpy(dOne,hOne,N*8,hipMemcpyHostToDevice);
	int blk=(N+255)/256;
	double *dFMul,*dFAdd; hipMalloc(&dFMul,N*8); hipMalloc(&dFAdd,N*8);
	k_fma<<<blk,256>>>(dA,dB,dZero,dM,dFMul,N);   // a*b
	k_fma<<<blk,256>>>(dA,dOne,dC,dM,dFAdd,N);    // a*1 + c = a+c
	k_div<<<blk,256>>>(dA,dB,dM,dD,N);
	k_sqrt<<<blk,256>>>(dA,dM,dS,N);
	hipDeviceSynchronize();
	double* gMul=new double[N]; double* gAdd=new double[N]; double* gD=new double[N]; double* gS=new double[N];
	hipMemcpy(gMul,dFMul,N*8,hipMemcpyDeviceToHost); hipMemcpy(gAdd,dFAdd,N*8,hipMemcpyDeviceToHost);
	hipMemcpy(gD,dD,N*8,hipMemcpyDeviceToHost); hipMemcpy(gS,dS,N*8,hipMemcpyDeviceToHost);

	int mM=0,mA=0,mD=0,mS=0;
	for(int i=0;i<N;i++){ if(u64(gMul[i])!=u64(cMul[i])) mM++; if(u64(gAdd[i])!=u64(cAdd[i])) mA++; if(u64(gD[i])!=u64(cD[i])) mD++; if(u64(gS[i])!=u64(cS[i])) mS++; }
	printf("MUL  EFT vs SSE oracle: %d / %d mismatches\n", mM, N);
	printf("ADD  EFT vs SSE oracle: %d / %d mismatches\n", mA, N);
	printf("DIV  EFT vs SSE oracle: %d / %d mismatches\n", mD, N);
	printf("SQRT EFT vs SSE oracle: %d / %d mismatches\n", mS, N);
	int s=0;
	for(int i=0;i<N&&s<5;i++){ if(u64(gMul[i])!=u64(cMul[i])){ printf("  MUL ex: a=%016llx b=%016llx m=%d g=%016llx cpu=%016llx\n",u64(hA[i]),u64(hB[i]),hM[i],u64(gMul[i]),u64(cMul[i])); s++; } }
  for(int i=0;i<N&&s<5;i++){ if(u64(gD[i])!=u64(cD[i])){ printf("  DIV ex: a=%016llx b=%016llx m=%d g=%016llx cpu=%016llx\n",u64(hA[i]),u64(hB[i]),hM[i],u64(gD[i]),u64(cD[i])); s++; } }
	s=0;
	for(int i=0;i<N&&s<5;i++){ if(u64(gS[i])!=u64(cS[i])){ printf("  SQRT ex: a=%016llx m=%d g=%016llx cpu=%016llx\n",u64(hA[i]),hM[i],u64(gS[i]),u64(cS[i])); s++; } }

	// restore MXCSR sanity
	_mm_setcsr(_mm_getcsr());
	delete[] hA;delete[] hB;delete[] hC;delete[] hM;delete[] hZero;delete[] hOne;delete[] cMul;delete[] cAdd;delete[] cD;delete[] cS;delete[] gMul;delete[] gAdd;delete[] gD;delete[] gS;
	hipFree(dA);hipFree(dB);hipFree(dC);hipFree(dM);hipFree(dZero);hipFree(dOne);hipFree(dFMul);hipFree(dFAdd);hipFree(dD);hipFree(dS);
	return 0;
}
