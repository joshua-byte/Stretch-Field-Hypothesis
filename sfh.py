# %%
import pandas as pd
import matplotlib.pyplot as plt

# Read the text file
df = pd.read_csv("data.txt", delim_whitespace=True, comment='#', header=None)

# Assign column names (adjust if needed)
df.columns = ["SN", "z", "mu", "mu_err", "extra_column"] # Added an extra column name

# Save as CSV
df.to_csv("Union2.1_mu_vs_z.csv", index=False)

# %%
df = pd.read_csv("data.csv")
df

# %%
df = df.dropna(subset=["z", "mu"])

# %%
plt.errorbar(df["z"], df["mu"], yerr=df["mu_err"], fmt='o', markersize=4, capsize=2, label="Union2.1 SN Ia")
plt.xlabel("Redshift z")
plt.ylabel("Distance Modulus μ")
plt.title("Union2.1 Supernovae Hubble Diagram")
plt.grid(True)
plt.legend()
plt.show()

# %%
import numpy as np
from scipy.integrate import quad

# Cosmology parameters
H0 = 70  # km/s/Mpc
c = 299792.458  # km/s
Omega_m = 0.3
Omega_L = 0.7

# Luminosity distance function
def E(z):
    return np.sqrt(Omega_m * (1+z)**3 + Omega_L)

def dL(z):
    integral, _ = quad(lambda z_prime: 1/E(z_prime), 0, z)
    return (1 + z) * (c / H0) * integral  # in Mpc

# Distance modulus μ
z_vals = np.linspace(0.01, max(df['z']), 500)
mu_model = 5 * np.log10([dL(z) for z in z_vals]) + 25

# Plot
plt.errorbar(df['z'], df['mu'], yerr=df['mu_err'], fmt='o', markersize=3, alpha=0.5, label="Supernova Data")
plt.plot(z_vals, mu_model, 'r-', lw=2, label="ΛCDM Model")
plt.xlabel("Redshift z")
plt.ylabel("Distance Modulus μ")
plt.title("Union2.1 Hubble Diagram with ΛCDM Model")
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 3e5  # km/s
H0 = 70  # Hubble constant km/s/Mpc

# Placeholder: your stretch field curvature function
def kappa_bar(z):
    return 0.1 * np.exp(-z)  # replace with your actual function

Omega_m = 0.3
Omega_r = 0.0
c_s = 1
k = 0

def H(z):
    a = 1 / (1 + z)
    H_squared = H0**2 * (Omega_m*(1+z)**3 + Omega_r*(1+z)**4) + (c_s**2/3) * kappa_bar(z) - k * (1/a**2)
    return np.sqrt(np.abs(H_squared))

def D_L(z):
    integral, _ = quad(lambda z_prime: c / H(z_prime), 0, z)
    return (1 + z) * integral

def mu(z):
    return 5 * np.log10(D_L(z)) + 25

# Load numeric columns only, skipping the supernova name
data = np.loadtxt('data.txt', usecols=(1,2,3))  # z, mu, mu_error
z_data = data[:,0]
mu_data = data[:,1]
mu_err = data[:,2]

mu_model = np.array([mu(z) for z in z_data])

plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='o', markersize=3, label='Union2.1 Data')
plt.plot(z_data, mu_model, 'r-', label='Stretch Field Model')
plt.xlabel('Redshift z')
plt.ylabel('Distance Modulus μ(z)')
plt.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ----------------------------
# Load Union2.1 data
# ----------------------------
# Skip the first column (SN name), keep numeric columns: z, mu, mu_err
data = np.genfromtxt('data.txt', delimiter='\t', dtype=float, usecols=(1,2,3))
z_data = data[:,0]
mu_data = data[:,1]
mu_err = data[:,2]

# ----------------------------
# Cosmological parameters
# ----------------------------
H0 = 70.0
c = 299792.458
Omega_m = 0.3
Omega_r = 0.0
Omega_k = 0.0
Omega_s = 0.7

def kappa(z):
    return 1.0  # replace with your dynamic bar-kappa(z)

def H(z):
    return H0 * np.sqrt(
        Omega_m*(1+z)**3 +
        Omega_r*(1+z)**4 +
        Omega_s*kappa(z) +
        Omega_k*(1+z)**2
    )

def d_L(z):
    integral, _ = quad(lambda zp: c / H(zp), 0, z)
    return (1+z) * integral

# ----------------------------
# Compute model μ(z)
# ----------------------------
z_model = np.linspace(0, 1.4, 300)
mu_model = np.array([5*np.log10(d_L(zp)*1e6 / 10) for zp in z_model])

# ----------------------------
# Plot
# ----------------------------
plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='o', color='blue', label='Union2.1 Data', alpha=0.6)
plt.plot(z_model, mu_model, color='red', lw=2, label='Stretch Field Model')
plt.xlabel('Redshift z')
plt.ylabel('Distance Modulus μ(z)')
plt.title('Union2.1 Data vs Stretch Field Model')
plt.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Physical constants (in suitable units)
c = 3e5  # km/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
H0 = 70  # km/s/Mpc, example Hubble constant

# Load Union2.1 data
data = np.genfromtxt('data.txt', dtype=None, encoding='utf-8', usecols=(1, 2, 3), names=['z', 'mu', 'mu_err'])
z_data = data['z']
mu_obs = data['mu']
mu_err = data['mu_err']

# Example Stretch Field Model parameters
rho_m0 = 0.3  # matter density parameter
rho_r0 = 0.0  # radiation density parameter, negligible here
cs2 = 1.0     # stretch field speed squared
k = 0.0       # curvature, assume flat

def H_z(z):
    """
    Hubble parameter as a function of redshift using Stretch Field Model.
    """
    a = 1/(1+z)
    kappa_bar = 0.0  # placeholder; replace with your actual kappa(t) function
    H2 = (8*np.pi*G/3)*(rho_m0 + rho_r0) + cs2/3*kappa_bar - k/a**2
    return np.sqrt(H2)

def luminosity_distance(z):
    """
    Compute D_L in Mpc via numerical integration
    """
    integrand = lambda zp: c / H0  # replace H0 with H_z(zp) once units are consistent
    d, _ = quad(integrand, 0, z)
    return (1+z)*d

def mu_model(z):
    """
    Distance modulus from Stretch Field Model
    """
    D_L = np.array([luminosity_distance(zi) for zi in z])
    return 5*np.log10(D_L) + 25  # D_L in Mpc

# Compute residuals
mu_pred = mu_model(z_data)
residuals = mu_obs - mu_pred

# Plot residuals
plt.errorbar(z_data, residuals, yerr=mu_err, fmt='o', color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('Residuals Δμ')
plt.title('Residuals: Union2.1 vs Stretch Field Model')
plt.savefig('residuals_union21.png')
plt.show()

# Compute RMS and reduced chi-squared
RMS = np.sqrt(np.mean(residuals**2))
chi2 = np.sum((residuals / mu_err)**2)
chi2_red = chi2 / (len(z_data) - 1)  # adjust if multiple parameters
print("RMS:", RMS, "Reduced chi2:", chi2_red)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Constants
c = 299792.458  # Speed of light in km/s

class CosmologicalModel:
    def __init__(self, H0=70.0, Omega_m=0.3, w=-1.0):
        """
        Initialize cosmological model parameters.

        Parameters:
        -----------
        H0 : float
            Hubble constant in km/s/Mpc
        Omega_m : float
            Matter density parameter
        w : float
            Dark energy equation of state parameter
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.w = w
        self.Omega_de = 1.0 - Omega_m  # Assuming flat universe

    def hubble_parameter(self, z):
        """
        Compute Hubble parameter H(z) for wCDM model.

        H(z)^2 = H0^2 * [Omega_m*(1+z)^3 + Omega_de*(1+z)^{3(1+w)}]
        """
        term1 = self.Omega_m * (1 + z)**3
        term2 = self.Omega_de * (1 + z)**(3 * (1 + self.w))
        return self.H0 * np.sqrt(term1 + term2)

    def comoving_distance_integrand(self, z):
        """Integrand for comoving distance calculation."""
        return c / self.hubble_parameter(z)

    def comoving_distance(self, z):
        """
        Calculate comoving distance to redshift z via numerical integration.
        """
        if np.isscalar(z):
            if z <= 0:
                return 0.0
            result, _ = quad(self.comoving_distance_integrand, 0, z)
            return result
        else:
            distances = []
            for zi in z:
                if zi <= 0:
                    distances.append(0.0)
                else:
                    result, _ = quad(self.comoving_distance_integrand, 0, zi)
                    distances.append(result)
            return np.array(distances)

    def luminosity_distance(self, z):
        """
        Calculate luminosity distance.
        D_L(z) = (1+z) * D_C(z)
        """
        return (1 + z) * self.comoving_distance(z)

    def distance_modulus(self, z):
        """
        Calculate distance modulus.
        mu(z) = 5 * log10(D_L(z)) + 25
        """
        D_L = self.luminosity_distance(z)
        return 5 * np.log10(D_L) + 25

def load_sne_data(filename='data.txt'):
    """Load Union2.1 SNe Ia data safely."""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

        # Skip first 5 header lines (alpha, beta, delta, M values)
        for line in lines[5:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    name = parts[0]
                    z = float(parts[1])
                    mu = float(parts[2])
                    mu_err = float(parts[3])
                    data.append([name, z, mu, mu_err])
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['name', 'z', 'mu', 'mu_err'])
    df[['z', 'mu', 'mu_err']] = df[['z', 'mu', 'mu_err']].astype(float)
    return df

def calculate_statistics(data_mu, model_mu, mu_err):
    """Calculate residuals, RMS, and reduced chi-squared."""
    residuals = data_mu - model_mu
    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals / mu_err)**2)
    dof = len(residuals) - 2  # 2 free parameters (Omega_m, w)
    reduced_chi2 = chi2 / dof

    return residuals, rms, chi2, reduced_chi2

def objective_function(params, z_data, mu_data, mu_err_data):
    """Objective function for parameter fitting."""
    Omega_m, w = params

    # Apply reasonable physical bounds
    if Omega_m <= 0 or Omega_m >= 1 or w < -3 or w > 1:
        return 1e10

    try:
        model = CosmologicalModel(H0=70.0, Omega_m=Omega_m, w=w)
        mu_model = model.distance_modulus(z_data)
        chi2 = np.sum(((mu_data - mu_model) / mu_err_data)**2)
        return chi2
    except:
        return 1e10

def fit_parameters(z_data, mu_data, mu_err_data, initial_guess=[0.3, -1.0]):
    """Fit cosmological parameters to minimize chi-squared."""
    print("Fitting cosmological parameters...")

    result = minimize(objective_function, initial_guess,
                     args=(z_data, mu_data, mu_err_data),
                     method='Nelder-Mead',
                     options={'maxiter': 1000, 'disp': True})

    return result

def create_publication_plot(df, model, fitted_params=None):
    """Create publication-ready Hubble diagram plot."""
    z_data = df['z'].values
    mu_data = df['mu'].values
    mu_err_data = df['mu_err'].values

    # Generate smooth redshift array for model curve
    z_smooth = np.logspace(np.log10(0.01), np.log10(1.5), 100)
    mu_model_smooth = model.distance_modulus(z_smooth)
    mu_model_data = model.distance_modulus(z_data)

    # Calculate residuals and statistics
    residuals, rms, chi2, reduced_chi2 = calculate_statistics(mu_data, mu_model_data, mu_err_data)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})

    # Main Hubble diagram
    ax1.errorbar(z_data, mu_data, yerr=mu_err_data, fmt='o',
                alpha=0.6, markersize=3, elinewidth=0.5, capsize=0,
                color='steelblue', label=f'Union2.1 SNe Ia (N={len(df)})')

    ax1.plot(z_smooth, mu_model_smooth, 'r-', linewidth=2,
            label=f'wCDM Model (Ωₘ={model.Omega_m:.3f}, w={model.w:.3f})')

    ax1.set_ylabel('Distance Modulus μ (mag)', fontsize=12)
    ax1.set_xlim(0.01, 1.5)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_title('Union2.1 SNe Ia Hubble Diagram with wCDM Model Fit', fontsize=14, fontweight='bold')

    # Add statistics text box
    stats_text = f'RMS = {rms:.3f} mag\nχ²/dof = {reduced_chi2:.2f}\nχ² = {chi2:.1f}'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Residuals subplot
    ax2.errorbar(z_data, residuals, yerr=mu_err_data, fmt='o',
                alpha=0.6, markersize=3, elinewidth=0.5, capsize=0, color='steelblue')
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax2.axhline(y=rms, color='red', linestyle='--', alpha=0.7, label=f'±RMS ({rms:.3f})')
    ax2.axhline(y=-rms, color='red', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Residuals Δμ (mag)', fontsize=12)
    ax2.set_xlim(0.01, 1.5)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    return fig, (rms, chi2, reduced_chi2)

# Main execution
if __name__ == "__main__":
    # Load the Union2.1 SNe Ia data
    print("Loading Union2.1 SNe Ia data...")
    df = load_sne_data('data.txt')
    print(f"Successfully loaded {len(df)} SNe Ia data points")
    print(f"Redshift range: {df['z'].min():.3f} - {df['z'].max():.3f}")

    # Extract data arrays
    z_data = df['z'].values
    mu_data = df['mu'].values
    mu_err_data = df['mu_err'].values

    # Initial model with standard ΛCDM parameters
    print("\n1. Computing initial ΛCDM model (w = -1.0)...")
    lambda_cdm = CosmologicalModel(H0=70.0, Omega_m=0.3, w=-1.0)

    # Calculate initial statistics
    mu_initial = lambda_cdm.distance_modulus(z_data)
    residuals_initial, rms_initial, chi2_initial, reduced_chi2_initial = calculate_statistics(
        mu_data, mu_initial, mu_err_data)

    print(f"Initial ΛCDM model statistics:")
    print(f"  RMS = {rms_initial:.4f} mag")
    print(f"  χ² = {chi2_initial:.2f}")
    print(f"  χ²/dof = {reduced_chi2_initial:.3f}")

    # Fit parameters to find best-fit wCDM model
    print("\n2. Fitting wCDM parameters...")
    fit_result = fit_parameters(z_data, mu_data, mu_err_data, initial_guess=[0.3, -1.0])

    if fit_result.success:
        best_Omega_m, best_w = fit_result.x
        print(f"\nBest-fit parameters:")
        print(f"  Ωₘ = {best_Omega_m:.4f}")
        print(f"  w = {best_w:.4f}")
        print(f"  χ² = {fit_result.fun:.2f}")

        # Create best-fit model
        best_fit_model = CosmologicalModel(H0=70.0, Omega_m=best_Omega_m, w=best_w)
    else:
        print("Optimization failed. Using initial parameters.")
        best_fit_model = lambda_cdm
        best_Omega_m, best_w = 0.3, -1.0

    # Create publication-ready plot
    print("\n3. Creating publication-ready Hubble diagram...")
    fig, (rms, chi2, reduced_chi2) = create_publication_plot(df, best_fit_model)

    # Print final results
    print(f"\nFinal Results:")
    print(f"  Model: wCDM with H₀ = 70 km/s/Mpc")
    print(f"  Best-fit Ωₘ = {best_Omega_m:.4f}")
    print(f"  Best-fit w = {best_w:.4f}")
    print(f"  RMS = {rms:.4f} mag")
    print(f"  χ² = {chi2:.2f}")
    print(f"  χ²/dof = {reduced_chi2:.3f}")
    print(f"  Degrees of freedom = {len(df) - 2}")

    # Save results to CSV
    results_df = df.copy()
    results_df['mu_model'] = best_fit_model.distance_modulus(z_data)
    results_df['residuals'] = mu_data - results_df['mu_model']
    results_df.to_csv('sne_analysis_results.csv', index=False)

    # Save plot
    plt.savefig('hubble_diagram_wcdm.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved to:")
    print(f"  - sne_analysis_results.csv (data with model predictions)")
    print(f"  - hubble_diagram_wcdm.png (publication-ready plot)")



