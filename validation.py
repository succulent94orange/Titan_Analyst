import numpy as np
import scipy.stats as stats

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def calculate_cohens_d(group1, group2):
    """Calculates Effect Size (magnitude of difference)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_se
    return d

def run_rigorous_validation():
    """
    Simulates a rigorous scientific process to validate a hypothesis 
    that 'Treatment increases Score'.
    """
    
    # --- STEP 1: DEFINE GROUND TRUTH (The "Universe") ---
    # We set a true effect size. In real life, you don't know this.
    # Here, Treatment is truly 0.5 SD better than Control.
    np.random.seed(42) # For reproducibility (Replication criterion)
    TRUE_EFFECT = 0.5 
    SAMPLE_SIZE = 1000 # Large sample size for '5-Sigma' potential
    
    print_header("1. EXPERIMENTAL SETUP (RCT)")
    print(f"Hypothesis: Treatment Group Mean > Control Group Mean")
    print(f"Sample Size: {SAMPLE_SIZE} per group (High Power)")
    
    # Generate Data
    # Control: Mean 50, SD 10
    control_group = np.random.normal(loc=50, scale=10, size=SAMPLE_SIZE)
    # Treatment: Mean 55 (50 + 0.5*10), SD 10
    treatment_group = np.random.normal(loc=50 + (TRUE_EFFECT * 10), scale=10, size=SAMPLE_SIZE)
    
    # --- STEP 2: STATISTICAL ARMOR (Z-Score & P-Value) ---
    print_header("2. STATISTICAL SIGNIFICANCE CHECK")
    
    t_stat, p_val = stats.ttest_ind(treatment_group, control_group, equal_var=False)
    
    # Calculate Sigma (Z-score approximation from p-value for two-tailed)
    # Norm.ppf(1 - p/2) gives the Z-score
    sigma_level = stats.norm.ppf(1 - p_val/2)
    
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_val:.20f}") # Print many decimals to show how small it is
    print(f"Sigma Level: {sigma_level:.2f}σ")
    
    if sigma_level >= 5:
        print(">> RESULT: 5-SIGMA REACHED. (Physics Standard of Discovery)")
    elif sigma_level >= 3:
        print(">> RESULT: 3-SIGMA REACHED. (Strong Evidence)")
    else:
        print(">> RESULT: Standard Significance only.")

    # --- STEP 3: MAGNITUDE CHECK (Effect Size) ---
    print_header("3. MAGNITUDE CHECK (Cohen's d)")
    
    d = calculate_cohens_d(treatment_group, control_group)
    print(f"Cohen's d: {d:.3f}")
    
    if d > 0.2:
        print(">> RESULT: Effect is practically significant (not just statistical).")
    else:
        print(">> RESULT: Effect is tiny. Might not matter in real life.")

    # --- STEP 4: UNCERTAINTY QUANTIFICATION (Bootstrapping) ---
    print_header("4. ROBUSTNESS CHECK (Bootstrapped 99% CI)")
    print("Resampling 10,000 times to simulate repeating the experiment...")
    
    n_iterations = 10000
    boot_diffs = []
    
    # Bootstrap loop
    for _ in range(n_iterations):
        # Sample with replacement (simulating new datasets)
        c_sample = np.random.choice(control_group, size=SAMPLE_SIZE, replace=True)
        t_sample = np.random.choice(treatment_group, size=SAMPLE_SIZE, replace=True)
        boot_diffs.append(np.mean(t_sample) - np.mean(c_sample))
    
    # 99% Confidence Interval (Stricter than standard 95%)
    ci_lower = np.percentile(boot_diffs, 0.5)
    ci_upper = np.percentile(boot_diffs, 99.5)
    
    print(f"Observed Difference: {np.mean(treatment_group) - np.mean(control_group):.3f}")
    print(f"99% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    if ci_lower > 0:
        print(">> RESULT: The 99% CI does not cross zero. Highly robust.")
    else:
        print(">> RESULT: CI crosses zero. We cannot be certain.")

    # --- STEP 5: FALSIFICATION (Permutation Test) ---
    print_header("5. FALSIFICATION ATTEMPT (Permutation Test)")
    print("Attempting to break the finding by shuffling labels...")
    print("If our result is real, shuffled data should rarely match our observed diff.")
    
    observed_diff = np.mean(treatment_group) - np.mean(control_group)
    combined = np.concatenate([control_group, treatment_group])
    fake_diffs = []
    n_permutations = 1000
    
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        # Split randomly
        fake_control = combined[:SAMPLE_SIZE]
        fake_treatment = combined[SAMPLE_SIZE:]
        fake_diffs.append(np.mean(fake_treatment) - np.mean(fake_control))
        
    fake_diffs = np.array(fake_diffs)
    # How many fake results were as extreme as our real result?
    p_perm = np.mean(fake_diffs >= observed_diff)
    
    print(f"Permutation P-value: {p_perm:.4f}")
    if p_perm == 0:
        print(">> RESULT: In 1000 simulations of random noise, NEVER saw this result.")
        print(">> CONCLUSION: It is extremely unlikely this is a fluke.")
    else:
        print(">> RESULT: We generated this result by chance.")

    # --- FINAL VERDICT ---
    print_header("FINAL VERDICT")
    if sigma_level > 3 and d > 0.2 and ci_lower > 0 and p_perm < 0.001:
        print("✅ HYPOTHESIS STATUS: BULLETPROOF")
        print("This finding has high power, high magnitude, passes stress testing,")
        print("and is virtually impossible to generate by random chance.")
    else:
        print("⚠️ HYPOTHESIS STATUS: TENTATIVE")

if __name__ == "__main__":
    run_rigorous_validation()
