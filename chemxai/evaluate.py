from data import prepare_data_graph
import numpy as np

class Evaluate_Tubular():
    
    def generate_pertubation(self, X, noise_levels=[0.01, 0.05, 0.1, 0.2], random_state=42):
        """
        Generate perturbed versions of the input data for robustness testing
        
        Parameters:
        -----------
        X : numpy.ndarray
            The original descriptors (e.g., Coulomb Matrix)
        noise_levels : list
            List of noise standard deviations to apply to features
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Dictionary containing the original data and perturbed versions
            Keys are noise levels (with '0' for original data)
        """
        np.random.seed(random_state)
        
        # Dictionary to store original and perturbed data
        robust_data = {'0': X}  # '0' key for the original data
        
        # Create perturbed versions with different noise levels
        for noise in noise_levels:
            # Generate Gaussian noise with specified std dev
            noise_matrix = np.random.normal(0, noise, size=X.shape)
            
            # Scale noise proportionally to the range of each feature
            feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
            scaled_noise = noise_matrix * feature_ranges
            
            # Add noise to the original data
            perturbed_data = X + scaled_noise
            
            # Store in dictionary with noise level as key
            robust_data[str(noise)] = perturbed_data
        
        return robust_data

    def generate_noise_feature(self, X, noise_type='gaussian', noise_scale=1.0, seed=42):
        """
        Generate a dataset with an additional noise feature column.
        This allows testing if explanation methods correctly identify the noise feature as unimportant.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Original feature matrix
        noise_type : str
            Type of noise to generate ('gaussian', 'uniform', or 'binary')
        noise_scale : float
            Scale parameter for the noise distribution
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple : (X_with_noise, feature_is_noise)
            - X_with_noise: Original X with added noise column
            - feature_is_noise: Boolean mask indicating which features are noise (True) vs real (False)
        """
        np.random.seed(seed)
        n_samples = X.shape[0]
        
        # Generate noise column based on specified type
        if noise_type == 'gaussian':
            noise_feature = np.random.normal(0, noise_scale, size=(n_samples, 1))
        elif noise_type == 'uniform':
            noise_feature = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, 1))
        elif noise_type == 'binary':
            noise_feature = np.random.choice([0, 1], size=(n_samples, 1))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Add noise feature to original data
        X_with_noise = np.hstack((X, noise_feature))
        
        # Create a mask to track which features are noise
        # False for real features, True for noise features
        feature_is_noise = np.zeros(X_with_noise.shape[1], dtype=bool)
        feature_is_noise[-1] = True  # Last column is the noise feature
        
        return X_with_noise, feature_is_noise


class Evaluate_Graph():
    pass


if __name__ == '__main__':
