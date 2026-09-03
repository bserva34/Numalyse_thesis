#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // REQUIS POUR NUMPY
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

// Convertit un cv::Mat (BGR) en array NumPy (H, W, 3) utilisable directement par YOLO
// Conversion sécurisée avec copie de tampon gérée par Python
py::array_t<unsigned char> mat_to_nparray(const cv::Mat& src) {
    // 1. On crée le conteneur NumPy à la bonne taille (H, W, Channels)
    py::array_t<unsigned char> dst({ src.rows, src.cols, src.channels() });
    
    // 2. On récupère un accès direct au tampon de destination NumPy
    auto mutable_hint = dst.mutable_unchecked();
    
    // 3. On copie proprement la mémoire ligne par ligne pour respecter le pas (stride) d'OpenCV
    for (int r = 0; r < src.rows; ++r) {
        std::memcpy(dst.mutable_data(r, 0, 0), src.ptr<unsigned char>(r), src.cols * src.channels());
    }
    
    return dst;
}

double double_gaussian_equal(double x, double t, double epsilon, double sigma) {
    double g1 = std::exp(-std::pow(x - t / 2.0, 2) / (2.0 * std::pow(sigma, 2)));
    double g2 = std::exp(-std::pow(x - (t - epsilon), 2) / (2.0 * std::pow(sigma, 2)));
    return g1 + g2;
}

double compute_sharpness(const cv::Mat& frame) {
    cv::Mat gray, laplacian;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

py::tuple process_video_cpp(
    const std::string& video_path,
    double thr,
    double sigma_pos,
    int num_classes,
    py::object python_model,
    int batch_size
) {
    cv::VideoCapture cap(video_path);
    int num_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    if (num_frames <= 0) {
        std::cerr << "Erreur : " << video_path << " ne contient aucune frame\n";
        return py::make_tuple(0.0, -1);
    }

    double video_duration = (fps > 0) ? (num_frames / fps) : 0.0;
    double epsilon = 0.1 * num_frames;
    double sigma_t = 0.15 * num_frames;

    std::vector<std::vector<float>> histograms(num_frames, std::vector<float>(num_classes, 0.0f));
    std::vector<double> sharpness_scores(num_frames, 0.0);

    std::vector<cv::Mat> batch_frames;
    std::vector<int> batch_indices;

    for (int i = 0; i < num_frames; ++i) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        batch_frames.push_back(frame.clone());
        batch_indices.push_back(i);

        if (batch_frames.size() == static_cast<size_t>(batch_size) || i == num_frames - 1) {
            py::list py_batch;
            for (const auto& f : batch_frames) {
                // Conversion propre en tableau Numpy pour éviter le crash C-API
                py_batch.append(mat_to_nparray(f)); 
            }

            // Inférence YOLO
            py::object outputs = python_model(py_batch, py::arg("verbose") = false);
            py::list out_list = outputs.cast<py::list>();

            for (size_t b = 0; b < batch_frames.size(); ++b) {
                int frame_idx = batch_indices[b];
                const auto& current_frame = batch_frames[b];
                py::object out = out_list[b];

                sharpness_scores[frame_idx] = compute_sharpness(current_frame);

                py::object boxes_obj = out.attr("boxes");
                
                // Extraction via tenseurs/numpy convertis en listes standards C++
                std::vector<float> confs = boxes_obj.attr("conf").attr("cpu")().attr("numpy")().cast<std::vector<float>>();
                std::vector<int> labels = boxes_obj.attr("cls").attr("cpu")().attr("numpy")().cast<std::vector<int>>();
                std::vector<std::vector<float>> boxes = boxes_obj.attr("xyxy").attr("cpu")().attr("numpy")().cast<std::vector<std::vector<float>>>();

                double H = current_frame.rows;
                double W = current_frame.cols;
                double cx_img = W / 2.0;
                double cy_img = H / 2.0;

                for (size_t d = 0; d < confs.size(); ++d) {
                    if (confs[d] <= thr) continue;

                    double cx = (boxes[d][0] + boxes[d][2]) / 2.0;
                    double cy = (boxes[d][1] + boxes[d][3]) / 2.0;

                    double dx = (cx - cx_img) / cx_img;
                    double dy = (cy - cy_img) / cy_img;
                    double dist = std::sqrt(dx * dx + dy * dy);

                    double w_pos = std::exp(-0.5 * std::pow(dist / sigma_pos, 2));
                    double weight = confs[d] * w_pos;

                    int label = labels[d];
                    if (label >= 0 && label < num_classes) {
                        histograms[frame_idx][label] += weight;
                    }
                }
            }
            batch_frames.clear();
            batch_indices.clear();
        }
    }
    cap.release();

    double max_sharpness = *std::max_element(sharpness_scores.begin(), sharpness_scores.end());
    if (max_sharpness > 0) {
        for (auto& s : sharpness_scores) s /= max_sharpness;
    }

    for (int i = 0; i < num_frames; ++i) {
        for (int c = 0; c < num_classes; ++c) {
            histograms[i][c] *= sharpness_scores[i];
        }
    }

    std::vector<double> weights_t(num_frames, 0.0);
    double sum_weights_t = 0.0;
    for (int i = 0; i < num_frames; ++i) {
        weights_t[i] = double_gaussian_equal(static_cast<double>(i), static_cast<double>(num_frames), epsilon, sigma_t);
        sum_weights_t += weights_t[i];
    }

    std::vector<double> hist_t(num_classes, 0.0);
    for (int i = 0; i < num_frames; ++i) {
        if (sum_weights_t > 0) weights_t[i] /= sum_weights_t;
        for (int c = 0; c < num_classes; ++c) {
            hist_t[c] += histograms[i][c] * weights_t[i];
        }
    }

    int id_frame = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < num_frames; ++i) {
        double dist = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            dist += std::pow(histograms[i][c] - hist_t[c], 2);
        }
        dist = std::sqrt(dist);

        if (dist < min_dist) {
            min_dist = dist;
            id_frame = i;
        }
    }

    return py::make_tuple(video_duration, id_frame);
}

PYBIND11_MODULE(video_processor_cxx, m) {
    m.def("process_video_cpp", &process_video_cpp, "Process video and find keyframe in C++");
}