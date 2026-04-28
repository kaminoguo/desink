"""Figure 1(c): Theorem 1 fit on Pythia-70M (R^2 = 0.978 with one parameter)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# DATA: B4 (norm_ratio = α proxy) + D1 (E1/ER raw & de-sinked), L3
# =============================================================================

# B4 v3: norm_ratio at L3 for each checkpoint
b4_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
b4_norm_ratio = [1.2386, 1.2385, 1.2386, 1.2373, 1.2093, 1.0611, 1.0716, 1.1365, 1.2134, 1.2417, 1.2428, 1.9450, 3.2844, 4.6828, 6.3338, 8.8487, 10.3075, 9.8898, 9.5947]

# D1: L3 metrics for each checkpoint
d1_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
d1_E1_raw = [0.0529, 0.0529, 0.0529, 0.0529, 0.0529, 0.0526, 0.0545, 0.2702, 0.2261, 0.1303, 0.0712, 0.0580, 0.0528, 0.2209, 0.3724, 0.5954, 0.6892, 0.7531, 0.7172, 0.6836]
d1_E1_ds  = [0.0425, 0.0425, 0.0425, 0.0425, 0.0427, 0.0434, 0.0455, 0.1327, 0.1641, 0.1151, 0.0663, 0.0429, 0.0384, 0.0426, 0.0539, 0.0652, 0.0733, 0.0716, 0.0738, 0.0750]
d1_ER_raw = [178.2, 178.2, 178.2, 178.2, 178.0, 176.9, 171.9, 60.9, 45.8, 78.7, 125.6, 175.6, 207.1, 119.8, 59.0, 17.3, 9.6, 6.4, 7.6, 9.2]
d1_ER_ds  = [191.3, 191.3, 191.3, 191.3, 191.1, 189.8, 184.9, 125.3, 70.1, 97.0, 138.0, 190.9, 224.2, 236.2, 231.6, 216.8, 199.6, 187.1, 161.0, 155.6]

# Align: B4 starts at step=1, D1 starts at step=0. 
# Map step 0 in D1 to step 1 in B4 (both are essentially init)
# Create aligned arrays using matching steps
b4_dict = dict(zip(b4_steps, b4_norm_ratio))
# For step 0 in D1, use step 1's norm_ratio from B4
b4_dict[0] = b4_dict[1]

aligned_steps = []
aligned_alpha = []
aligned_E1_raw = []
aligned_E1_ds = []
aligned_ER_raw = []
aligned_ER_ds = []

for i, s in enumerate(d1_steps):
    if s in b4_dict:
        aligned_steps.append(s)
        aligned_alpha.append(b4_dict[s])
        aligned_E1_raw.append(d1_E1_raw[i])
        aligned_E1_ds.append(d1_E1_ds[i])
        aligned_ER_raw.append(d1_ER_raw[i])
        aligned_ER_ds.append(d1_ER_ds[i])

aligned_alpha = np.array(aligned_alpha)
aligned_E1_raw = np.array(aligned_E1_raw)
aligned_E1_ds = np.array(aligned_E1_ds)
aligned_ER_raw = np.array(aligned_ER_raw)
aligned_ER_ds = np.array(aligned_ER_ds)

print(f"Aligned {len(aligned_steps)} checkpoints")
print(f"Alpha range: {aligned_alpha.min():.2f} to {aligned_alpha.max():.2f}")
print(f"E1_raw range: {aligned_E1_raw.min():.4f} to {aligned_E1_raw.max():.4f}")
print(f"ER_raw range: {aligned_ER_raw.min():.1f} to {aligned_ER_raw.max():.1f}")

# =============================================================================
# MATHEMATICAL MODEL
# =============================================================================
# 
# Model: X = α * s * 1^T + X_c
# where s is unit sink direction, α is sink magnitude
# 
# Covariance: C = α² * s*s^T + C_c  (assuming s ⊥ content, approximately)
# 
# E1 share ≈ (α² + λ_1^c) / (α² + Tr(C_c))
# where λ_1^c is the largest eigenvalue of C_c
#
# If content is roughly isotropic with total variance V_c spread over d dimensions:
# E1 ≈ (α² + V_c/d) / (α² + V_c)  ... but this assumes content E1 = 1/d
#
# More generally, let e1_c = true content E1 share, V_c = total content variance
# Then: λ_1^c = e1_c * V_c
# E1_raw ≈ (α² + e1_c * V_c) / (α² + V_c)
#
# But α here is norm_ratio, which is a ratio, not absolute magnitude.
# Actually norm_ratio = ||h_pos0|| / mean(||h_other||)
# The sink's contribution to variance scales as (norm_ratio)^2 relative to content
#
# So let's define: r = norm_ratio, and model:
# E1_raw = (r² * k + e1_c) / (r² * k + 1)
# where k is a scaling constant and e1_c is the de-sinked E1 (true content E1)
#
# For ER, the effective rank:
# ER = exp(H) where H is the entropy of normalized eigenvalue spectrum
# This is harder to get closed-form, but approximately:
# When sink dominates, ER → 1
# When sink is absent, ER = ER_c (content effective rank)
# Interpolation: ER_raw ≈ ER_c / (1 + k * r²) ... too crude
#
# Better: think of it as a 2-component spectrum
# One eigenvalue = α² + small, rest = content eigenvalues
# Normalized: p_1 = (α² + ...) / (α² + V_c), p_i = λ_i^c / (α² + V_c)
# As α grows, p_1 → 1, all p_i → 0, entropy → 0, ER → 1

# =============================================================================
# FIT: E1 model
# =============================================================================

# Model: E1_raw(r) = (k * r² + e1_c(t)) / (k * r² + 1)
# Problem: e1_c varies with training step (it's the de-sinked E1)
# So this isn't a simple 1-parameter fit of r → E1

# Better approach: predict E1_raw from (norm_ratio, E1_ds)
# E1_raw_predicted = (k * r² + E1_ds) / (k * r² + 1)
# Only one free parameter: k

def E1_model(params, r, e1_ds):
    """E1_raw = (k * r^2 + e1_ds) / (k * r^2 + 1)"""
    k = params[0]
    return (k * r**2 + e1_ds) / (k * r**2 + 1)

def E1_residuals(params, r, e1_ds, e1_raw):
    return E1_model(params, r, e1_ds) - e1_raw

from scipy.optimize import least_squares
# Fit k
result = least_squares(E1_residuals, x0=[0.1], args=(aligned_alpha, aligned_E1_ds, aligned_E1_raw))
k_fit = result.x[0]
E1_predicted = E1_model(result.x, aligned_alpha, aligned_E1_ds)
E1_residual = aligned_E1_raw - E1_predicted
E1_r2 = 1 - np.sum(E1_residual**2) / np.sum((aligned_E1_raw - aligned_E1_raw.mean())**2)

print(f"\n=== E1 Model ===")
print(f"k = {k_fit:.4f}")
print(f"R² = {E1_r2:.6f}")
print(f"Max |residual| = {np.max(np.abs(E1_residual)):.4f}")
print(f"Mean |residual| = {np.mean(np.abs(E1_residual)):.4f}")

# Print per-checkpoint comparison
print(f"\n{'Step':>8} {'α':>6} {'E1_ds':>7} {'E1_raw':>7} {'E1_pred':>7} {'resid':>7}")
for i in range(len(aligned_steps)):
    print(f"{aligned_steps[i]:>8} {aligned_alpha[i]:>6.2f} {aligned_E1_ds[i]:>7.4f} {aligned_E1_raw[i]:>7.4f} {E1_predicted[i]:>7.4f} {E1_residual[i]:>+7.4f}")


# =============================================================================
# FIT: ER model  
# =============================================================================
# For effective rank, the relationship is more complex
# ER depends on the full spectrum, not just E1
# 
# Approximate model: The sink adds one dominant eigenvalue proportional to α²
# This concentrates the spectrum, reducing ER
#
# A simple model: ER_raw ≈ ER_ds / (1 + k_er * (r² - 1))
# or: 1/ER_raw ≈ 1/ER_ds + k_er * r²
# 
# Actually, let's think about it differently.
# The normalized spectrum with sink: p_1 ≈ (k*r² + e1_ds), p_i ≈ p_i^c * (1 - k*r²/(k*r²+1))
# This means all content eigenvalues get suppressed by factor 1/(k*r²+1)
# And the new dominant eigenvalue gets fraction (k*r² + e1_ds)/(k*r²+1)
#
# For the entropy:
# H_raw = -p_1*log(p_1) - sum_{i>1} p_i*log(p_i)
# 
# Let's try a simpler phenomenological model:
# ER_raw = 1 + (ER_ds - 1) / (1 + k_er * (r-1)²)
# This gives ER_raw → 1 as r → ∞, and ER_raw = ER_ds when r = 1

def ER_model(params, r, er_ds):
    """ER_raw = 1 + (ER_ds - 1) / (1 + k * (r-1)^2)"""
    k = params[0]
    return 1 + (er_ds - 1) / (1 + k * (r - 1)**2)

def ER_residuals(params, r, er_ds, er_raw):
    return ER_model(params, r, er_ds) - er_raw

result_er = least_squares(ER_residuals, x0=[1.0], args=(aligned_alpha, aligned_ER_ds, aligned_ER_raw))
k_er_fit = result_er.x[0]
ER_predicted = ER_model(result_er.x, aligned_alpha, aligned_ER_ds)
ER_residual = aligned_ER_raw - ER_predicted
ER_r2 = 1 - np.sum(ER_residual**2) / np.sum((aligned_ER_raw - aligned_ER_raw.mean())**2)

print(f"\n=== ER Model ===")
print(f"k_er = {k_er_fit:.4f}")
print(f"R² = {ER_r2:.6f}")
print(f"Max |residual| = {np.max(np.abs(ER_residual)):.1f}")
print(f"Mean |residual| = {np.mean(np.abs(ER_residual)):.1f}")

print(f"\n{'Step':>8} {'α':>6} {'ER_ds':>7} {'ER_raw':>7} {'ER_pred':>7} {'resid':>7}")
for i in range(len(aligned_steps)):
    print(f"{aligned_steps[i]:>8} {aligned_alpha[i]:>6.2f} {aligned_ER_ds[i]:>7.1f} {aligned_ER_raw[i]:>7.1f} {ER_predicted[i]:>7.1f} {ER_residual[i]:>+7.1f}")


# =============================================================================
# Also try: use k from E1 fit for a "principled" ER model
# =============================================================================
# If E1_raw = (k*r² + e1_ds) / (k*r² + 1), then the sink captures fraction
# f_sink = k*r² / (k*r² + 1) of total variance
# And content captures 1/(k*r² + 1) of total variance
# 
# For a two-component model where one eigenvalue has weight f_sink + e1_ds*(1-f_sink)
# and the rest have the de-sinked spectrum rescaled by (1-f_sink):
# 
# Actually, let's try the most principled version.
# Let the de-sinked spectrum have eigenvalues {λ_i} summing to 1 (normalized)
# After adding sink: new eigenvalues are {f_sink + λ_1*(1-f_sink), λ_2*(1-f_sink), ...}
# where f_sink = k*r²/(k*r²+1)
#
# ER = exp(-sum p_i log p_i)
# This requires knowing the full de-sinked spectrum, which we don't have.
# But we can estimate: if de-sinked ER = ER_ds, the de-sinked spectrum has entropy H_ds = log(ER_ds)
#
# After adding sink with fraction f:
# New spectrum: p_1 = f + (1-f)*q_1, p_i = (1-f)*q_i for i>1
# where {q_i} is the de-sinked normalized spectrum
# H_new = -p_1*log(p_1) - (1-f)*sum_{i>1} q_i * log((1-f)*q_i)
#        = -p_1*log(p_1) - (1-f)*[sum_{i>1} q_i*log(q_i) + sum_{i>1} q_i*log(1-f)]
#        = -p_1*log(p_1) - (1-f)*[H_ds_tail + (1-q_1)*log(1-f)]
# where H_ds_tail = -sum_{i>1} q_i*log(q_i) = H_ds + q_1*log(q_1)
#
# This is getting complicated. Let's just verify the phenomenological fit is good and move on.

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nE1 model: E1_raw = (k·r² + E1_ds) / (k·r² + 1)")
print(f"  k = {k_fit:.4f}, R² = {E1_r2:.4f}")
print(f"\nER model: ER_raw = 1 + (ER_ds - 1) / (1 + k_er·(r-1)²)")  
print(f"  k_er = {k_er_fit:.4f}, R² = {ER_r2:.4f}")

if E1_r2 > 0.95 and ER_r2 > 0.90:
    print(f"\n✅ THEORY-EMPIRICAL FIT IS STRONG. Safe to write the math section.")
elif E1_r2 > 0.90 and ER_r2 > 0.80:
    print(f"\n⚠️  Fit is decent but not perfect. May need to discuss deviations.")
else:
    print(f"\n❌ Fit is weak. Need to revise the model before writing.")

# =============================================================================
# CROSS-VALIDATION: Use D4 data to check if model generalizes
# =============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION ON D4 (Pythia-6.9B, GPT-2-XL)")
print("="*80)

# For D4, we don't have norm_ratio per layer, but we can back-calculate
# From E1 model: k*r² = E1_raw*(k*r²+1) - E1_ds = ... 
# Actually we can check: given E1_raw and E1_ds, does k*r² = (E1_raw - E1_ds)/(1 - E1_raw)?
# If so, the implied r should be physically reasonable

print("\nPythia-6.9B: Implied k*r² from E1 model")
p6_E1_raw = [0.165, 0.931, 0.949, 0.929, 0.901, 0.853, 0.777, 0.700, 0.075]
p6_E1_ds  = [0.035, 0.030, 0.024, 0.023, 0.025, 0.036, 0.047, 0.051, 0.059]
p6_ER_raw = [388.3, 2.1, 1.8, 2.2, 2.8, 4.2, 7.5, 13.4, 460.4]
p6_ER_ds  = [736.9, 1098.1, 1298.4, 1302.1, 1199.1, 960.4, 787.3, 745.9, 568.3]
p6_layers = ['L0', 'L4', 'L8', 'L12', 'L16', 'L20', 'L24', 'L28', 'L31']

print(f"{'Layer':>5} {'E1_raw':>7} {'E1_ds':>6} {'kr²':>8} {'implied_r':>10}")
for i in range(len(p6_layers)):
    if p6_E1_raw[i] < 0.999:  # avoid division by zero
        kr2 = (p6_E1_raw[i] - p6_E1_ds[i]) / (1 - p6_E1_raw[i])
        implied_r = np.sqrt(kr2 / k_fit) if kr2 > 0 else 0
        print(f"{p6_layers[i]:>5} {p6_E1_raw[i]:>7.3f} {p6_E1_ds[i]:>6.3f} {kr2:>8.2f} {implied_r:>10.1f}")
        
        # Check ER prediction
        if implied_r > 0:
            er_pred = ER_model([k_er_fit], implied_r, p6_ER_ds[i])
            print(f"       ER_raw={p6_ER_raw[i]:.1f}, ER_pred={er_pred:.1f}")
