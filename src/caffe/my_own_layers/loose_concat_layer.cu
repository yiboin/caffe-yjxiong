#include <vector>

#include "caffe/my_own_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LooseConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if(bottom.size() == 1) { 
    return; 
  }

  /// here only support `num` or `channel` axis 
  CHECK_GE(concat_axis_, 0) << "here only support `num` or `channel` axis...";
  CHECK_LE(concat_axis_, 1) << "here only support `num` or `channel` axis...";

  // 0: along with `num` axis | 1: along with `channel`
  const bool flag = concat_axis_ == 0 ? true : false;

  int n2 = 0;
  int c2 = 0;

  Dtype* top_data = top[0]->mutable_gpu_data();
  // ??? set zero before concatenation ???
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

  // start & copy -- data
  for(int i = 0; i < bottom.size(); ++i) {
    const int copy_size      = bottom[i]->width();
    const Dtype* bottom_data = bottom[i]->gpu_data();

    if(flag) {
      for(int n = 0; n < bottom[i]->num(); n++) {
        for(int c = 0; c < bottom[i]->channels(); c++) {
          for(int h = 0; h < bottom[i]->height(); h++) {
            caffe_copy(
                copy_size, 
                bottom_data + bottom[i]->offset(n, c, h), 
                top_data + top[0]->offset(n2 + n, c, h)
            );
          }
        }
      }

      // offset for `num` axis
      n2 += bottom[i]->num();

    } else {
      for(int n = 0; n < bottom[i]->num(); n++) {
        for(int c = 0; c < bottom[i]->channels(); c++) {
          for(int h = 0; h < bottom[i]->height(); h++) {
            caffe_copy(
                copy_size, 
                bottom_data + bottom[i]->offset(n, c, h), 
                top_data + top[0]->offset(n, c2 + c, h)
            );
          }
        }
      }

      // offset for `channel` axis
      c2 += bottom[i]->channels();
    }
  } // end outer loop
}

template <typename Dtype>
void LooseConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(bottom.size() == 1) { 
    return; 
  }

  /// here only support `num` or `channel` axis 
  CHECK_GE(concat_axis_, 0) << "here only support `num` or `channel` axis...";
  CHECK_LE(concat_axis_, 1) << "here only support `num` or `channel` axis...";

  // 0: along with `num` axis | 1: along with `channel`
  const bool flag = concat_axis_ == 0 ? true : false;

  int n2 = 0;
  int c2 = 0;
  const Dtype* top_diff = top[0]->gpu_diff();

  // start & copy -- diff
  for(int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) {
      if(flag) {
        // offset for `num` axis
        n2 += bottom[i]->num();
      } else {
        // offset for `channel` axis
        c2 += bottom[i]->channels();
      }
      continue;
    }

    const int copy_size = bottom[i]->width();
    // ??? set zero ???
    Dtype* bottom_diff  = bottom[i]->mutable_gpu_diff();
    caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom_diff);

    if(flag) {
      for(int n = 0; n < bottom[i]->num(); n++) {
        for(int c = 0; c < bottom[i]->channels(); c++) {
          for(int h = 0; h < bottom[i]->height(); h++) {
            caffe_copy(
                copy_size, 
                top_diff + top[0]->offset(n2 + n, c, h),
                bottom_diff + bottom[i]->offset(n, c, h)
            );
          }
        }
      }

      // offset for `num` axis
      n2 += bottom[i]->num();

    } else {
      for(int n = 0; n < bottom[i]->num(); n++) {
        for(int c = 0; c < bottom[i]->channels(); c++) {
          for(int h = 0; h < bottom[i]->height(); h++) {
            caffe_copy(
                copy_size, 
                top_diff + top[0]->offset(n, c2 + c, h),
                bottom_diff + bottom[i]->offset(n, c, h)
            );
          }
        }
      }

      // offset for `channel` axis
      c2 += bottom[i]->channels();
    }
  } // end outer loop
}

INSTANTIATE_LAYER_GPU_FUNCS(LooseConcatLayer);

}  // namespace caffe
