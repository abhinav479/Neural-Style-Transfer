import tensorflow as tf
import numpy as np
import PIL.Image

class NeuralStyleTransfer:
    def __init__(self, content_layers=None, style_layers=None):
        if content_layers is None:
            content_layers = ['block5_conv2']
        if style_layers is None:
            style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        self.vgg = self.vgg_layers(self.style_layers + self.content_layers)
        self.vgg.trainable = False

    def load_img(self, path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def vgg_layers(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def style_content_loss(self, outputs, style_targets, content_targets):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= 1e-2 / self.num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= 1e4 / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    def style_content_model(self):
        return StyleContentModel(self.vgg, self.style_layers, self.content_layers, self.gram_matrix)

    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def run(self, content_path, style_path, output_path, epochs=2, steps_per_epoch=100, st_progress_bar=None):
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        extractor = self.style_content_model()
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = self.style_content_loss(outputs, style_targets, content_targets)
            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, 0.0, 1.0))
            return loss

        image = tf.Variable(content_image)
        all_losses = []

        total_steps = epochs * steps_per_epoch
        step_counter = 0

        for n in range(epochs):
            for m in range(steps_per_epoch):
                loss = train_step(image)
                all_losses.append(loss.numpy())
                step_counter += 1
                if st_progress_bar is not None:
                    st_progress_bar.progress(step_counter / total_steps)

        result_image = self.tensor_to_image(image)
        result_image.save(output_path)

        return all_losses


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, vgg, style_layers, content_layers, gram_matrix):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.gram_matrix = gram_matrix

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
