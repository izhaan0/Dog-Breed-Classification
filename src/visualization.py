"""
Visualization module for Dog Breed Classification project
"""

import numpy as np
import matplotlib.pyplot as plt


def show_25_images(images, labels, unique_breeds):
    """Display a plot of 25 images and their labels from a data batch"""
    plt.figure(figsize=(10, 10))
    
    for i in range(25):
        # Create subplots (5 rows, 5 columns)
        ax = plt.subplot(5, 5, i+1)
        
        # Display image
        plt.imshow(images[i])
        
        # Add image label as title
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def get_pred_label(prediction_probabilities, unique_breeds):
    """Get predicted label from prediction probabilities"""
    return unique_breeds[np.argmax(prediction_probabilities)]


def plot_pred(prediction_probabilities, labels, images, unique_breeds, n=1):
    """Plot prediction results for a single image"""
    pred_prob = prediction_probabilities[n]
    true_label = labels[n]
    image = images[n]
    
    # Get predicted label
    pred_label = get_pred_label(pred_prob, unique_breeds)
    
    # Plot image
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    
    # Set color based on correctness
    color = "green" if pred_label == true_label else "red"
    
    # Set title
    plt.title(f"{pred_label} {np.max(pred_prob)*100:.0f}% ({true_label})", color=color)


def plot_pred_conf(prediction_probabilities, labels, unique_breeds, n=1):
    """Plot prediction confidence for top 10 predictions"""
    pred_prob = prediction_probabilities[n]
    true_label = labels[n]
    
    pred_label = get_pred_label(pred_prob, unique_breeds)
    
    # Get top 10 predictions
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    
    # Create bar plot
    top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                      top_10_pred_values, 
                      color="grey")
    
    plt.xticks(np.arange(len(top_10_pred_labels)), 
              labels=top_10_pred_labels, 
              rotation="vertical")
    
    # Highlight true label if in top 10
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")


def plot_predictions_grid(prediction_probabilities, labels, images, unique_breeds, 
                         start_idx=0, num_rows=3, num_cols=2):
    """Plot a grid of predictions with confidence bars"""
    num_images = num_rows * num_cols
    plt.figure(figsize=(10*num_cols, 5*num_rows))
    
    for i in range(num_images):
        # Plot prediction image
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_pred(prediction_probabilities, labels, images, unique_breeds, 
                 n=i+start_idx)
        
        # Plot confidence bars
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_pred_conf(prediction_probabilities, labels, unique_breeds, 
                      n=i+start_idx)
    
    plt.tight_layout(h_pad=1.0)
    plt.show()


def plot_custom_predictions(custom_images, custom_pred_labels):
    """Plot custom image predictions"""
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(custom_images):
        plt.subplot(1, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(custom_pred_labels[i])
        plt.imshow(image)
