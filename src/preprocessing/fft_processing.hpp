#ifndef FFT_PROCESSING_H
#define FFT_PROCESSING_H

#include "datatypes.hpp"

namespace fftProcessing
{
	void apply_hilbert_transform_samples(RadarData::Frame& frame);
	void apply_fft1(RadarData::Frame& frame);
	void apply_fft2(RadarData::Frame& frame);
	void fft(std::vector<std::complex<double>>& data, bool inverse = false);
	void fftProcessPipeline(RadarData::Frame& frame);
	void apply_hanning_window(std::vector<std::complex<double>>& data);
	void normalize_fft_output(std::vector<std::complex<double>>& data, size_t fft_length);
}

#endif