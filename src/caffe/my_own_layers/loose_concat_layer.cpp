#include <vector>

#include "caffe/my_own_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LooseConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void LooseConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes                  = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
    if(concat_axis_ < 0) {
      concat_axis_ += num_axes;
    }
  }
  
  // Don't allow negative indexing for concat_dim, a uint32 -- almost
  // certainly unintended.
  CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
      << "produced negative result; concat_dim must satisfy "
      << "0 <= concat_dim < " << kMaxBlobAxes;
  CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";

  // Initialize with the first blob.
  vector<int> top_shape  = bottom[0]->shape();
  vector<int> top_shape2 = bottom[0]->shape();
  num_concats_           = bottom[0]->count(0, concat_axis_);
  concat_input_size_     = bottom[0]->count(concat_axis_ + 1);
  int bottom_count_sum   = bottom[0]->count();

  for(int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes()) << "All inputs must have the same #axes.";

    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { 
        continue; 
      }
      /// loose concat
      top_shape2[j] = std::max(top_shape[j], bottom[i]->shape(j));
    }

    bottom_count_sum         += bottom[i]->count();
    top_shape[concat_axis_]  += bottom[i]->shape(concat_axis_);
    top_shape2[concat_axis_] += bottom[i]->shape(concat_axis_);
  }

  top[0]->Reshape(top_shape2);
  CHECK_LE(bottom_count_sum, top[0]->count());

  if (bottom.size() == 1) {
    CHECK_EQ(top[0]->count(), bottom[0]->count()) << "must equal...";
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void LooseConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  Dtype* top_data = top[0]->mutable_cpu_data();
  // ??? set zero before concatenation ???
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // start & copy -- data
  for(int i = 0; i < bottom.size(); ++i) {
    const int copy_size      = bottom[i]->width();
    const Dtype* bottom_data = bottom[i]->cpu_data();

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
void LooseConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype* top_diff = top[0]->cpu_diff();

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
    Dtype* bottom_diff  = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);

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

#ifdef CPU_ONLY
STUB_GPU(LooseConcatLayer);
#endif

INSTANTIATE_CLASS(LooseConcatLayer);
REGISTER_LAYER_CLASS(LooseConcat);

}  // namespace caffe
