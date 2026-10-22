import sys
import os
import io
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Output directory for plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

print("=" * 70)
print("POME: MATHEMATICAL MODELING OF STUDENT AI WILLINGNESS-TO-PAY & DEMAND")
print("=" * 70)

# 1. Load and Clean Dataset
df = pd.read_excel('Data.xlsx', sheet_name='Synthetic Responses')
N = len(df)
print(f"Total respondents: N = {N}")

# Mapping functions
q5_map = {'Not at all useful': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Extremely useful': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q5_num'] = df['Q5_AI_Usefulness'].map(q5_map).astype(float)

q6_map = {'Never': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Very frequently': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q6_num'] = df['Q6_Free_Tool_Limitations'].map(q6_map).astype(float)

q9_map = {'Not dependent at all': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Extremely dependent': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q9_num'] = df['Q9_AI_Dependence'].map(q9_map).astype(float)

q27_map = {'Very Unlikely': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Very Likely': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q27_num'] = df['Q27_Likelihood_to_Pay_Next_12_Months'].map(q27_map).astype(float)

q3_map = {'Never': 0, 'Rarely': 1, 'A few times a month': 2, 'A few times a week': 3, 'Once a day': 4, 'Multiple times a day': 5}
df['Q3_num'] = df['Q3_AI_Use_Frequency'].map(q3_map).astype(float)

q8_map = {'I rarely/never use AI': 0.0, 'Less than 15 minutes': 0.15, '15-30 minutes': 0.375, '30-60 minutes': 0.75, '1-2 hours': 1.5, 'More than 2 hours': 2.5}
df['Q8_hours'] = df['Q8_Daily_AI_Time'].map(q8_map).astype(float)

q11_map = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5}
df['Q11_num'] = df['Q11_Free_Tool_Failure'].map(q11_map).astype(float)

# Willingness to pay midpoint
q18_midpoint = {
    'I would not pay': 0.0,
    'Less than ₹100': 50.0,
    '₹100–₹199': 150.0,
    '₹200–₹299': 250.0,
    '₹300–₹399': 350.0,
    '₹400–₹499': 450.0,
    '₹500–₹699': 600.0,
    '₹700 or more': 800.0
}
df['WTP_midpoint'] = df['Q18_Max_Willingness_to_Pay'].map(q18_midpoint).astype(float)

# Seniority & Branches
df['Seniority'] = df['Q1_Year_of_Study'].str.extract(r'(\d+)').astype(float)
df['Branch_CS_AI'] = df['Q2_Engineering_Branch'].apply(lambda x: 1 if any(k in str(x) for k in ['Computer Science', 'Artificial Intelligence', 'Information Science']) else 0)
df['Branch_ECE_EEE'] = df['Q2_Engineering_Branch'].apply(lambda x: 1 if any(k in str(x) for k in ['Electronics', 'Electrical']) else 0)
df['Coding_Purpose'] = df['Q4_Main_Purpose'].apply(lambda x: 1 if 'Coding' in str(x) else 0)
df['Is_Current_Payer'] = df['Q14_Current_Payment_Status'].apply(lambda x: 1 if 'Yes' in str(x) else 0)
df['High_Pay_Intent'] = ((df['Q27_num'] >= 4) | (df['Is_Current_Payer'] == 1)).astype(int)

# Expected subscription duration in months
q25_months = {
    'I would not subscribe': 0.0,
    'Less than 1 month': 0.5,
    'Only temporarily when I need it': 1.0,
    '1–3 months': 2.0,
    '4–6 months': 5.0,
    '7–12 months': 9.5,
    'More than 1 year': 15.0
}
df['Expected_Duration_Months'] = df['Q25_Expected_Subscription_Duration'].map(q25_months).fillna(0.0)

results = {}

# ==============================================================================
# MODEL 1: GABOR-GRANGER & ECONOMETRIC DEMAND MODELS
# ==============================================================================
print("\n" + "="*50)
print("SECTION 1: DEMAND ESTIMATION & REVENUE MAXIMIZATION")
print("="*50)

price_points = np.array([99.0, 199.0, 399.0, 599.0])
cols_q17 = ['Q17_₹99_per_month', 'Q17_₹199_per_month', 'Q17_₹399_per_month', 'Q17_₹599_per_month']

top2_demand = []
weighted_demand = []
def_yes_demand = []

for p, col in zip(price_points, cols_q17):
    counts = df[col].value_counts(normalize=True)
    d_yes = counts.get('Definitely Yes', 0.0)
    p_yes = counts.get('Probably Yes', 0.0)
    n_sure = counts.get('Not Sure', 0.0)
    
    # Conversion metrics
    top2 = d_yes + p_yes
    weighted = 0.75 * d_yes + 0.25 * p_yes + 0.10 * n_sure
    
    top2_demand.append(top2)
    weighted_demand.append(weighted)
    def_yes_demand.append(d_yes)

top2_demand = np.array(top2_demand)
weighted_demand = np.array(weighted_demand)
def_yes_demand = np.array(def_yes_demand)

print(f"Prices: {price_points}")
print(f"Top-2 Box Purchase Intent (Q_top2): {np.round(top2_demand, 4)}")
print(f"Standard Weighted Intent (Q_weighted): {np.round(weighted_demand, 4)}")

demand_fits = {}

for name, Q_obs in [('Top2_Box', top2_demand), ('Weighted', weighted_demand)]:
    print(f"\n--- Fitting Demand Models for: {name} ---")
    
    # 1. Linear Model: Q = a + b*P
    lin_res = stats.linregress(price_points, Q_obs)
    a_lin, b_lin = lin_res.intercept, lin_res.slope
    r2_lin = lin_res.rvalue**2
    p_opt_lin = -a_lin / (2 * b_lin)
    q_opt_lin = a_lin + b_lin * p_opt_lin
    rev_opt_lin = p_opt_lin * q_opt_lin
    
    # 2. Power Model: ln Q = ln a - eps * ln P
    log_p = np.log(price_points)
    log_q = np.log(Q_obs)
    pow_res = stats.linregress(log_p, log_q)
    ln_a_pow, eps_pow = pow_res.intercept, -pow_res.slope
    a_pow = np.exp(ln_a_pow)
    r2_pow = pow_res.rvalue**2
    
    # 3. Exponential Model: ln Q = ln a - b * P
    exp_res = stats.linregress(price_points, log_q)
    ln_a_exp, b_exp = exp_res.intercept, -exp_res.slope
    a_exp = np.exp(ln_a_exp)
    r2_exp = exp_res.rvalue**2
    p_opt_exp = 1.0 / b_exp
    q_opt_exp = a_exp * np.exp(-b_exp * p_opt_exp)
    rev_opt_exp = p_opt_exp * q_opt_exp
    
    # 4. Logistic Model: logit(Q) = ln(Q / (1 - Q)) = alpha + beta * P
    # Q(P) = 1 / (1 + exp(-(alpha + beta * P)))
    logit_q = np.log(Q_obs / (1.0 - Q_obs))
    logit_res = stats.linregress(price_points, logit_q)
    alpha_logit, beta_logit = logit_res.intercept, logit_res.slope
    q_pred_logit = 1.0 / (1.0 + np.exp(-(alpha_logit + beta_logit * price_points)))
    r2_logit = r2_score(Q_obs, q_pred_logit)
    
    # Numerical optimization for Logistic Revenue: R(P) = P / (1 + exp(-(alpha + beta * P)))
    def neg_rev_logit(p):
        q = 1.0 / (1.0 + np.exp(-(alpha_logit + beta_logit * p)))
        return - (p * q)
    
    opt_logit = optimize.minimize_scalar(neg_rev_logit, bounds=(10, 2000), method='bounded')
    p_opt_logit = opt_logit.x
    q_opt_logit = 1.0 / (1.0 + np.exp(-(alpha_logit + beta_logit * p_opt_logit)))
    rev_opt_logit = -opt_logit.fun
    
    fit_summary = {
        'Linear': {
            'formula': f"Q(P) = {a_lin:.4f} - {-b_lin:.6f} * P",
            'a': float(a_lin), 'b': float(b_lin),
            'R2': float(r2_lin),
            'P_optimal': float(p_opt_lin),
            'Q_at_P_opt': float(q_opt_lin),
            'Max_Expected_Rev_per_Student': float(rev_opt_lin),
            'Elasticity_formula': f"Ed(P) = {b_lin:.6f} * (P / Q(P))"
        },
        'Exponential': {
            'formula': f"Q(P) = {a_exp:.4f} * exp(-{b_exp:.6f} * P)",
            'a': float(a_exp), 'b': float(b_exp),
            'R2': float(r2_exp),
            'P_optimal': float(p_opt_exp),
            'Q_at_P_opt': float(q_opt_exp),
            'Max_Expected_Rev_per_Student': float(rev_opt_exp),
            'Elasticity_formula': f"Ed(P) = -{b_exp:.6f} * P"
        },
        'Power_Constant_Elasticity': {
            'formula': f"Q(P) = {a_pow:.4f} * P^(-{eps_pow:.4f})",
            'a': float(a_pow), 'epsilon': float(eps_pow),
            'R2': float(r2_pow),
            'Price_Elasticity': float(-eps_pow)
        },
        'Logistic': {
            'formula': f"Q(P) = 1 / (1 + exp(-({alpha_logit:.4f} - {-beta_logit:.6f} * P)))",
            'alpha': float(alpha_logit), 'beta': float(beta_logit),
            'R2': float(r2_logit),
            'P_optimal': float(p_opt_logit),
            'Q_at_P_opt': float(q_opt_logit),
            'Max_Expected_Rev_per_Student': float(rev_opt_logit)
        }
    }
    demand_fits[name] = fit_summary
    
    print(f"  Linear Model:      {fit_summary['Linear']['formula']}, R^2 = {r2_lin:.4f}, P* = ₹{p_opt_lin:.2f}, Max Rev = ₹{rev_opt_lin:.2f}")
    print(f"  Exponential Model: {fit_summary['Exponential']['formula']}, R^2 = {r2_exp:.4f}, P* = ₹{p_opt_exp:.2f}, Max Rev = ₹{rev_opt_exp:.2f}")
    print(f"  Power Model:       {fit_summary['Power_Constant_Elasticity']['formula']}, R^2 = {r2_pow:.4f}, Elasticity = {-eps_pow:.4f}")
    print(f"  Logistic Model:    {fit_summary['Logistic']['formula']}, R^2 = {r2_logit:.4f}, P* = ₹{p_opt_logit:.2f}, Max Rev = ₹{rev_opt_logit:.2f}")

results['demand_models'] = demand_fits

# ==============================================================================
# MODEL 2: MULTIVARIATE LOGISTIC REGRESSION (DISCRETE CHOICE MODEL)
# ==============================================================================
print("\n" + "="*50)
print("SECTION 2: ECONOMETRIC DISCRETE CHOICE MODEL (LOGISTIC REGRESSION)")
print("="*50)

logit_formula = (
    "High_Pay_Intent ~ Seniority + Branch_CS_AI + Branch_ECE_EEE + "
    "Q3_num + Q8_hours + Q5_num + Q6_num + Q9_num + Q11_num + Coding_Purpose"
)
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
print(logit_model.summary())

params = logit_model.params
conf_int = logit_model.conf_int()
pvalues = logit_model.pvalues
odds_ratios = np.exp(params)
odds_ci_lower = np.exp(conf_int[0])
odds_ci_upper = np.exp(conf_int[1])

logit_df = pd.DataFrame({
    'Coefficient': params,
    'Std_Error': logit_model.bse,
    'z_stat': logit_model.tvalues,
    'p_value': pvalues,
    'Odds_Ratio': odds_ratios,
    'OR_95_CI_Lower': odds_ci_lower,
    'OR_95_CI_Upper': odds_ci_upper
})

pred_probs = logit_model.predict(df)
pred_class = (pred_probs >= 0.5).astype(int)
auc_score = roc_auc_score(df['High_Pay_Intent'], pred_probs)
conf_mat = confusion_matrix(df['High_Pay_Intent'], pred_class)
tn, fp, fn, tp = conf_mat.ravel()
accuracy = (tp + tn) / N
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

mcfadden_r2 = logit_model.prsquared
aic = logit_model.aic
bic = logit_model.bic
llf = logit_model.llf
llnull = logit_model.llnull
lr_chi2 = 2 * (llf - llnull)
lr_pval = stats.chi2.sf(lr_chi2, logit_model.df_model)

results['logistic_regression'] = {
    'summary_table': logit_df.to_dict(orient='index'),
    'metrics': {
        'pseudo_r2': float(mcfadden_r2),
        'lr_chi2': float(lr_chi2),
        'lr_pvalue': float(lr_pval),
        'aic': float(aic),
        'bic': float(bic),
        'auc': float(auc_score),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    }
}

# ==============================================================================
# MODEL 3: WILLINGNESS-TO-PAY SURVIVAL ANALYSIS & VALUATION DISTRIBUTION
# ==============================================================================
print("\n" + "="*50)
print("SECTION 3: WTP SURVIVAL ANALYSIS & VALUATION DISTRIBUTION")
print("="*50)

sorted_wtp = np.sort(df['WTP_midpoint'].values)
prices_eval = np.linspace(0, 800, 161)
emp_survival = np.array([np.mean(sorted_wtp >= p) for p in prices_eval])

valid_idx = (prices_eval > 0) & (emp_survival > 0) & (emp_survival < 1)
p_fit = prices_eval[valid_idx]
s_fit = emp_survival[valid_idx]

# Log-logistic fit: log((1 - S)/S) = beta * log(P) - beta * log(alpha)
y_ll = np.log((1.0 - s_fit) / s_fit)
x_ll = np.log(p_fit)
ll_res = stats.linregress(x_ll, y_ll)
beta_ll = ll_res.slope
alpha_ll = np.exp(-ll_res.intercept / beta_ll)
r2_surv_ll = ll_res.rvalue**2

# Weibull fit: ln(-ln(S)) = k * ln(P) - k * ln(lambda)
y_wb = np.log(-np.log(s_fit))
wb_res = stats.linregress(x_ll, y_wb)
k_wb = wb_res.slope
lambda_wb = np.exp(-wb_res.intercept / k_wb)
r2_surv_wb = wb_res.rvalue**2

wtp_mean = np.mean(df['WTP_midpoint'])
wtp_median = np.median(df['WTP_midpoint'])
wtp_q25 = np.percentile(df['WTP_midpoint'], 25)
wtp_q75 = np.percentile(df['WTP_midpoint'], 75)

rev_surv_emp = prices_eval * emp_survival
best_p_idx = np.argmax(rev_surv_emp)
best_p_surv = prices_eval[best_p_idx]
max_rev_surv = rev_surv_emp[best_p_idx]

results['wtp_survival'] = {
    'empirical_stats': {
        'mean': float(wtp_mean),
        'median': float(wtp_median),
        'q25': float(wtp_q25),
        'q75': float(wtp_q75),
        'iqr': float(wtp_q75 - wtp_q25)
    },
    'log_logistic_fit': {
        'alpha_scale': float(alpha_ll),
        'beta_shape': float(beta_ll),
        'r2': float(r2_surv_ll),
        'formula': f"S(P) = 1 / (1 + (P / {alpha_ll:.2f})^{beta_ll:.4f})"
    },
    'weibull_fit': {
        'lambda_scale': float(lambda_wb),
        'k_shape': float(k_wb),
        'r2': float(r2_surv_wb),
        'formula': f"S(P) = exp(-(P / {lambda_wb:.2f})^{k_wb:.4f})"
    },
    'optimal_pricing_from_survival': {
        'P_star': float(best_p_surv),
        'Expected_Revenue': float(max_rev_surv),
        'Penetration_at_P_star': float(emp_survival[best_p_idx])
    }
}

# ==============================================================================
# MODEL 4: SEGMENTATION & HETEROGENEITY ANALYSIS
# ==============================================================================
print("\n" + "="*50)
print("SECTION 4: SEGMENTATION & DEMAND HETEROGENEITY")
print("="*50)

cs_wtp = df[df['Branch_CS_AI'] == 1]['WTP_midpoint']
non_cs_wtp = df[df['Branch_CS_AI'] == 0]['WTP_midpoint']
mwu_res = stats.mannwhitneyu(cs_wtp, non_cs_wtp, alternative='two-sided')
ttest_res = stats.ttest_ind(cs_wtp, non_cs_wtp)

heavy_wtp = df[df['Q3_num'] >= 4]['WTP_midpoint']
occasional_wtp = df[df['Q3_num'] < 4]['WTP_midpoint']
mwu_heavy = stats.mannwhitneyu(heavy_wtp, occasional_wtp, alternative='two-sided')

results['segmentation'] = {
    'cs_vs_non_cs': {
        'cs_n': int(len(cs_wtp)),
        'cs_mean_wtp': float(cs_wtp.mean()),
        'cs_median_wtp': float(cs_wtp.median()),
        'non_cs_n': int(len(non_cs_wtp)),
        'non_cs_mean_wtp': float(non_cs_wtp.mean()),
        'non_cs_median_wtp': float(non_cs_wtp.median()),
        'mwu_pvalue': float(mwu_res.pvalue),
        't_pvalue': float(ttest_res.pvalue)
    },
    'heavy_vs_occasional': {
        'heavy_n': int(len(heavy_wtp)),
        'heavy_mean_wtp': float(heavy_wtp.mean()),
        'occasional_n': int(len(occasional_wtp)),
        'occasional_mean_wtp': float(occasional_wtp.mean()),
        'mwu_pvalue': float(mwu_heavy.pvalue)
    }
}

# ==============================================================================
# MODEL 5: CUSTOMER LIFETIME VALUE (CLV) & RETENTION MODEL
# ==============================================================================
print("\n" + "="*50)
print("SECTION 5: CUSTOMER LIFETIME VALUE (CLV) & RETENTION MODEL")
print("="*50)

marginal_cost = 40.0
discount_rate = 0.01

clv_scenarios = []
# Top2 box conversion probabilities
a_top = demand_fits['Top2_Box']['Linear']['a']
b_top = demand_fits['Top2_Box']['Linear']['b']

for p in [99.0, 149.0, 199.0, 249.0, 299.0, 399.0, 499.0, 599.0]:
    q_top = max(0.0, float(a_top + b_top * p))
    # Churn increases with price (from Q21)
    churn_rate = min(0.60, 0.16 + 0.00045 * p)
    retention_months = 1.0 / churn_rate
    gross_margin_monthly = max(0.0, p - marginal_cost)
    clv_subscriber = gross_margin_monthly / (churn_rate + discount_rate)
    expected_clv_per_market_student = q_top * clv_subscriber
    
    clv_scenarios.append({
        'Price': p,
        'Conversion_Rate_Top2': q_top,
        'Monthly_Gross_Margin': gross_margin_monthly,
        'Expected_Duration_Months': retention_months,
        'Subscriber_CLV': clv_subscriber,
        'Expected_CLV_per_Student': expected_clv_per_market_student
    })

clv_df = pd.DataFrame(clv_scenarios)
results['clv_model'] = {
    'marginal_cost': marginal_cost,
    'discount_rate_monthly': discount_rate,
    'scenarios': clv_scenarios
}

with open('model_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print("Model results saved to model_results.json")

# ==============================================================================
# GENERATE VISUALIZATIONS
# ==============================================================================
print("\nGenerating charts...")

# Figure 1: Demand Curves & Revenue Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
p_grid = np.linspace(50, 700, 200)

ax1.scatter(price_points, top2_demand, color='#2b5c8f', s=70, zorder=5, label='Empirical Top-2 Intent (Def + Prob Yes)')
ax1.scatter(price_points, weighted_demand, color='#c0392b', s=70, zorder=5, label='Empirical Weighted Intent')

lin_top = demand_fits['Top2_Box']['Linear']['a'] + demand_fits['Top2_Box']['Linear']['b'] * p_grid
log_a = demand_fits['Top2_Box']['Logistic']['alpha']
log_b = demand_fits['Top2_Box']['Logistic']['beta']
log_top = 1.0 / (1.0 + np.exp(-(log_a + log_b * p_grid)))

ax1.plot(p_grid, lin_top, color='#27ae60', lw=2.2, label=f"Linear Fit ($R^2={demand_fits['Top2_Box']['Linear']['R2']:.3f}$)")
ax1.plot(p_grid, log_top, color='#e67e22', lw=2.0, linestyle='--', label=f"Logistic Fit ($R^2={demand_fits['Top2_Box']['Logistic']['R2']:.3f}$)")

ax1.set_title('Demand Curve: Price vs. Adoption Probability', fontsize=12, fontweight='bold')
ax1.set_xlabel('Monthly Subscription Price (INR)', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Adoption Probability Q(P)', fontsize=11, fontweight='semibold')
ax1.set_ylim(0, 0.8)
ax1.set_xlim(50, 700)
ax1.legend(frameon=True, fontsize=9)

rev_lin = p_grid * lin_top
rev_log = p_grid * log_top
ax2.plot(p_grid, rev_lin, color='#27ae60', lw=2.2, label='Expected Rev (Linear Demand)')
ax2.plot(p_grid, rev_log, color='#e67e22', lw=2.0, linestyle='--', label='Expected Rev (Logistic Demand)')

p_opt_lin = demand_fits['Top2_Box']['Linear']['P_optimal']
rev_opt_lin = demand_fits['Top2_Box']['Linear']['Max_Expected_Rev_per_Student']

ax2.axvline(p_opt_lin, color='#27ae60', linestyle='-.', alpha=0.7)
ax2.scatter([p_opt_lin], [rev_opt_lin], color='#27ae60', s=90, zorder=6)
ax2.annotate(f"Optimal Price:\nP* = INR {p_opt_lin:.0f}\nMax Rev = INR {rev_opt_lin:.1f}/user",
             xy=(p_opt_lin, rev_opt_lin), xytext=(p_opt_lin - 160, rev_opt_lin - 30),
             arrowprops=dict(arrowstyle='->', lw=1.2, color='#27ae60'),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60'))

ax2.set_title('Expected Revenue Curve R(P) = P * Q(P)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Monthly Subscription Price (INR)', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Expected Revenue per Student (INR/month)', fontsize=11, fontweight='semibold')
ax2.set_xlim(50, 700)
ax2.legend(frameon=True, fontsize=9)
plt.subplots_adjust(wspace=0.25, left=0.08, right=0.95, top=0.90, bottom=0.12)
plt.savefig(os.path.join(plot_dir, 'fig1_demand_and_revenue.png'))
plt.close()

# Figure 2: Price Elasticity
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
b_val = demand_fits['Top2_Box']['Linear']['b']
ed_lin = b_val * p_grid / np.maximum(lin_top, 0.01)

ax.plot(p_grid, ed_lin, color='#2980b9', lw=2.5, label='Linear Demand Elasticity Ed(P)')
ax.axhline(-1.0, color='#c0392b', linestyle='--', lw=1.5, label='Unit Elasticity Threshold (Ed = -1.0)')
ax.fill_between(p_grid, 0, -1.0, color='#27ae60', alpha=0.12, label='Inelastic Zone (|Ed| < 1, Price Hikes Increase Revenue)')
ax.fill_between(p_grid, -1.0, -4.0, color='#e74c3c', alpha=0.10, label='Elastic Zone (|Ed| > 1, Volume Drops Dominate)')

ax.set_title('Price Elasticity of Demand Across Price Points', fontsize=12, fontweight='bold')
ax.set_xlabel('Monthly Price (INR)', fontsize=11, fontweight='semibold')
ax.set_ylabel('Price Elasticity Ed', fontsize=11, fontweight='semibold')
ax.set_xlim(50, 700)
ax.set_ylim(-3.5, 0.2)
ax.legend(frameon=True, fontsize=9, loc='lower left')
plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
plt.savefig(os.path.join(plot_dir, 'fig2_price_elasticity.png'))
plt.close()

# Figure 3: Logistic Regression Odds Ratios (Forest Plot)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
features_to_plot = [f for f in logit_df.index if f != 'Intercept']
clean_feature_names = {
    'Seniority': 'Academic Seniority (Year 1 to 4)',
    'Branch_CS_AI': 'Branch: Computer Science & AI',
    'Branch_ECE_EEE': 'Branch: Electronics & Electrical',
    'Q3_num': 'AI Usage Frequency (Scale 0 to 5)',
    'Q8_hours': 'Daily AI Time (Hours/Day)',
    'Q5_num': 'Perceived AI Usefulness (Scale 1 to 5)',
    'Q6_num': 'Free Limitation Frustration (Scale 1 to 5)',
    'Q9_num': 'AI Dependence (Scale 1 to 5)',
    'Q11_num': 'Free Tool Failure Rate (Scale 1 to 5)',
    'Coding_Purpose': 'Primary Purpose: Coding & Dev'
}

y_pos = np.arange(len(features_to_plot))
ors = [logit_df.loc[f, 'Odds_Ratio'] for f in features_to_plot]
ci_lows = [logit_df.loc[f, 'OR_95_CI_Lower'] for f in features_to_plot]
ci_highs = [logit_df.loc[f, 'OR_95_CI_Upper'] for f in features_to_plot]
labels = [clean_feature_names.get(f, f) for f in features_to_plot]

ax.errorbar(ors, y_pos, xerr=[np.array(ors) - np.array(ci_lows), np.array(ci_highs) - np.array(ors)],
            fmt='o', color='#2c3e50', ecolor='#95a5a6', elinewidth=1.8, capsize=4, markersize=7)
for i in range(len(y_pos)):
    c = '#27ae60' if ors[i] > 1.0 else '#c0392b'
    ax.scatter(ors[i], y_pos[i], color=c, s=70, zorder=5)

ax.axvline(1.0, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Odds Ratio (with 95% Confidence Interval)', fontsize=11, fontweight='semibold')
ax.set_title('Discrete Choice Predictors of High Willingness to Pay', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(0.4, 3.5)
plt.subplots_adjust(left=0.38, right=0.95, top=0.90, bottom=0.12)
plt.savefig(os.path.join(plot_dir, 'fig3_logistic_odds_ratios.png'))
plt.close()

# Figure 4: WTP Survival Function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

ax1.step(prices_eval, emp_survival, where='post', color='#2c3e50', lw=2.2, label='Empirical Survival P(WTP >= P)')
ax1.plot(prices_eval, 1.0 / (1.0 + (prices_eval / alpha_ll)**beta_ll), color='#e67e22', lw=2.0, linestyle='--', label=f'Log-Logistic Model (R^2={r2_surv_ll:.3f})')
ax1.plot(prices_eval, np.exp(-(prices_eval / lambda_wb)**k_wb), color='#27ae60', lw=1.8, linestyle=':', label=f'Weibull Model (R^2={r2_surv_wb:.3f})')

ax1.axvline(wtp_median, color='#c0392b', linestyle='-.', lw=1.3, label=f'Median WTP = INR {wtp_median:.0f}')
ax1.axvline(wtp_mean, color='#2980b9', linestyle='-.', lw=1.3, label=f'Mean WTP = INR {wtp_mean:.0f}')

ax1.set_title('WTP Survival Function: S(P) = P(WTP >= P)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Price Threshold P (INR)', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Probability Willing to Pay >= P', fontsize=11, fontweight='semibold')
ax1.set_xlim(0, 800)
ax1.set_ylim(0, 1.05)
ax1.legend(frameon=True, fontsize=9)

cs_surv = np.array([np.mean(cs_wtp >= p) for p in prices_eval])
non_cs_surv = np.array([np.mean(non_cs_wtp >= p) for p in prices_eval])

ax2.plot(prices_eval, cs_surv, color='#2980b9', lw=2.2, label=f'CS / AI Branches (Mean=INR {cs_wtp.mean():.0f})')
ax2.plot(prices_eval, non_cs_surv, color='#e74c3c', lw=2.2, linestyle='--', label=f'Non-CS Branches (Mean=INR {non_cs_wtp.mean():.0f})')

ax2.set_title('Segment Demand Comparison: CS/AI vs. Non-CS', fontsize=12, fontweight='bold')
ax2.set_xlabel('Monthly Price (INR)', fontsize=11, fontweight='semibold')
ax2.set_ylabel('Adoption Probability S(P)', fontsize=11, fontweight='semibold')
ax2.set_xlim(0, 800)
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=True, fontsize=9)

plt.subplots_adjust(wspace=0.25, left=0.08, right=0.95, top=0.90, bottom=0.12)
plt.savefig(os.path.join(plot_dir, 'fig4_wtp_survival_and_segments.png'))
plt.close()

# Figure 5: CLV Curve
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
prices_clv = clv_df['Price']
ax.plot(prices_clv, clv_df['Subscriber_CLV'], color='#2980b9', lw=2.2, marker='o', label='Per-Subscriber Lifetime Value (INR)')
ax.plot(prices_clv, clv_df['Expected_CLV_per_Student'] * 4, color='#27ae60', lw=2.5, marker='s', label='Market Expected CLV (scaled x4)')

best_clv_row = clv_df.loc[clv_df['Expected_CLV_per_Student'].idxmax()]
ax.axvline(best_clv_row['Price'], color='#e67e22', linestyle='--', lw=1.5)
ax.annotate(f"Optimal Long-Term Price:\nP = INR {best_clv_row['Price']:.0f}\nMax Market CLV",
            xy=(best_clv_row['Price'], best_clv_row['Expected_CLV_per_Student'] * 4),
            xytext=(best_clv_row['Price'] - 120, best_clv_row['Expected_CLV_per_Student'] * 4 + 100),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='#e67e22'),
            fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#27ae60'))

ax.set_title('Customer Lifetime Value (CLV) vs. Subscription Price', fontsize=12, fontweight='bold')
ax.set_xlabel('Monthly Subscription Price (INR)', fontsize=11, fontweight='semibold')
ax.set_ylabel('Lifetime Value (INR)', fontsize=11, fontweight='semibold')
ax.legend(frameon=True, fontsize=9, loc='upper left')
plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
plt.savefig(os.path.join(plot_dir, 'fig5_clv_curve.png'))
plt.close()

print("All charts generated and saved successfully!")
