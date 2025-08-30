# Report:

## Attack methods:

Report on the Jacobian-based Saliency Map Attack (JSMA)

1. Introduction:

- This report provides a detailed analysis of the Jacobian-based Saliency Map Attack (JSMA), a white-box adversarial attack method. JSMA is a targeted and sparse attack, aiming for minimal visible changes.
- It uses an iterative approach, stops either when the attack succeeds or when a maximum number of perturbations ,controlled by the gamma parameter, is reached.

2. Attack Methodology:

- The JSMA function operates by iteratively modifying a single pixel at a time, based on the calculated saliency map. The core of the attack revolves around three principles:

* White-Box Access:

- The attack is a "white-box" method, meaning it requires full knowledge of the target model's architecture and parameters. This is essential for calculating the Jacobian matrix, which is the foundation of the attack.

* The Jacobian Matrix:

- The Jacobian matrix is a tensor of first-order partial derivatives. In this context, it represents how the output logits change with respect to each pixel. The algorithm uses this matrix to identify which pixels, when changed, will have the most significant impact on the model's output.

* Saliency Map Calculation:

- The saliency map is the central heuristic of the JSMA. For a targeted attack ,where the goal is to misclassify an image as a specific target class, the saliency map for each pixel is calculated using the gradients of two specific logits:

    1. The gradient of the target class's logit with respect to the pixel.

    2. The sum of the gradients of all other classes' logits with respect to the same pixel.

- The algorithm prioritizes pixels that, when altered, simultaneously increase the score of the target class and decrease the scores of all other classes. The saliency map value for a pixel is positive only if it satisfies these conditions, and the pixel with the highest positive value is chosen for modification.

3. Key Parameters:

- theta: A small positive or negative value representing the amount of perturbation to be added to a pixel.

- gamma: A scalar between 0 and 1 that limits the maximum percentage of pixels that can be modified. This parameter ensures the attack remains subtle and sparse.

4. Conclusion:

The JSMA function, as implemented, provides a clear and effective demonstration of a sparse, targeted adversarial attack. It highlights how minor, carefully selected perturbations can significantly alter a neural network's prediction. The attack serves as a valuable case study for understanding model vulnerabilities and the importance of adversarial robustness in machine learning.

## Model explainability:

1. Occlusion based sailency maps:

    * Occlusion is a permutation based method for explaining machine learning models, by systematically masking regions of the image and observing how the output will change compared to the original images output.
    * The difference between the baseline prediction and the occluded prediction is called the importance score, which is added to all k x k pixels in the salency map, with red colors meaning higher importance score.
    * Finally, the salency map values are normalized, as the scores will be biased, so we divide the pixel's score with the total number of times it has been occluded.
    * This method can be used generally with some hyper-parameters:
        - Patch size(k x k).            
        - Stride.
        - Patch value: what will be placed in each pixel instead of its original value(mean, random, grey).
    * This process is computationaly expensive and does not take into account the global context.

2. Gradient-weighted Class Activation Mapping (Grad-CAM):

    * It used to explain CNNs, by highlighting pixels/ regions that are important for a given classification. 
    * Grad-CAM produces heatmaps called Class activation Maps(CAMs), which show parts that are used to classify the image as a particular class.
    * It weights feature maps in a models convulational layer using class gradients.
    * This method is more efficient and faster, but it mainly works on CNN for classification and it only shows the pixels in a specific region are contributing to the classification.

    * We need to specify the last convolutional layer to Grad-CAM, in our case it is the layer called layer4 in ResNet-34.

## Defense Method:

- 