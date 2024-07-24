#include "./grad_utils.h"

namespace raf {
namespace op {
namespace grad {

RAF_OP_GRAD("raf.op.one_hot", NoGrads<3>);

}  // namespace grad
}  // namespace op
}  // namespace raf
