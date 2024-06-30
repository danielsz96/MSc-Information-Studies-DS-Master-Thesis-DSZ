import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explain_shap(X_test, data, model):

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    shap_explainer = shap.DeepExplainer(model.model, X_test_tensor)

    shap_values = shap_explainer.shap_values(X_test_tensor).squeeze()

    temporal_shap_values = np.sum(shap_values[:, 6:], axis=1, keepdims=True)
    aggregated_shap_values = np.hstack((shap_values[:, :6], temporal_shap_values))

    feature_names = list(data.drop(columns=['bpd_label']).columns[:6]) + ['temporal_data']
    feature_names = [str(f) for f in feature_names]

    shap.summary_plot(np.expand_dims(aggregated_shap_values[0], 0), np.expand_dims(X_test[0, :7], 0),
                      feature_names=np.array(feature_names), plot_type = 'bar', max_display = 20)
    


def compute_gradcam(model, x):

    X_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    model.model.zero_grad()
    output = model.model(X_tensor)

    loss = torch.sum(output)
    loss.backward()

    gradients = model.tcn.gradients
    activations = model.tcn.activations

    batch_size, n_features, seq_len = gradients.size()
    grad_cam_values = torch.zeros(batch_size, n_features, seq_len)

    for i in range(n_features):
        weights = torch.mean(gradients[:, i, :], dim=1, keepdim=True)
        grad_cam_values[:, i, :] = weights * activations[:, i, :]

    grad_cam_values_min = grad_cam_values.min(dim=2, keepdim=True)[0]
    grad_cam_values_max = grad_cam_values.max(dim=2, keepdim=True)[0]
    grad_cam_values = (grad_cam_values - grad_cam_values_min) / (grad_cam_values_max - grad_cam_values_min + 1e-10)

    return grad_cam_values.detach().numpy()


def plot_gradcam_heatmap(grad_cam_values, n_features, original_input, grad_cam_inter=7):

    labels = [f'day_{i+1}' for i in range(grad_cam_inter)]

    grad_cam_values = torch.nn.functional.interpolate(grad_cam_values, size = grad_cam_inter, mode = 'linear')
    grad_cam_values = grad_cam_values / grad_cam_values.sum(dim=2, keepdim=True)
    plt.figure(figsize=(12, 4))

    for sample in range(grad_cam_values.shape[0]):

        for channel in range(n_features):
            plt.subplot(grad_cam_values.shape[0] * 2, n_features, sample * 2 * n_features + channel + 1)
            plt.plot(original_input.reshape(-1))
            plt.title(f"Sample {sample+1}, Input Channel: LSTM Encoded Temporal")

        for channel in range(n_features):
            plt.subplot(grad_cam_values.shape[0] * 2, n_features, sample * 2 * n_features + n_features + channel + 1)
            sns.heatmap(grad_cam_values[sample, channel, :].reshape(1, -1), cmap="viridis", cbar=False,
                        yticklabels=False, xticklabels=labels, annot=True)
            plt.title(f"Sample {sample+1}, Grad-CAM Channel: LSTM Encoded Temporal")

    plt.tight_layout()
    plt.show()