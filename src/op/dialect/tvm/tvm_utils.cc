/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_utils.cc
 * \brief Implementation of utility methods for TVM dialect.
 */
#include <cstdlib>
#include "raf/value.h"
#include "raf/registry.h"
#include "./tvm_utils.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::value;
using namespace raf::ir;
using namespace raf::registry;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;

MetaPersistCache<TVMModuleCacheEntry> CacheBuildCpu("tvm_cpu");
MetaPersistCache<TVMModuleCacheEntry> CacheBuildCuda("tvm_cuda");
MetaPersistCache<RelayFuncCacheEntry> CacheLoweredFunc("tvm_lower");

Value GetFakeValue(const Type& type, const Device& dev) {
  FakeValueCreator creator(dev);
  return creator(type);
}

void GetDLTensor(const Value& v, std::vector<DLTensor>* tensors) {
  if (v->IsInstance<TensorValueObj>()) {
    DLTensor* t = v;
    tensors->emplace_back(*t);
  } else if (const auto* tv = v.as<TupleValueObj>()) {
    for (const auto& v : tv->fields) {
      DLTensor* t = v;
      tensors->emplace_back(*t);
    }
  } else {
    LOG(FATAL) << "InternalError: TVMOpEnv does not deal with " << v->GetTypeKey();
    throw;
  }
}

Type GetTensorType(const DLTensor& dlt) {
  auto shape = GetShape<Integer>(dlt);
  return TensorType({shape.begin(), shape.end()}, ir::DataType(dlt.dtype));
}

Type GetTupleType(const std::vector<DLTensor>& dlts) {
  std::vector<Type> types;
  for (const auto& dlt : dlts) {
    types.emplace_back(GetTensorType(dlt));
  }
  return TupleType(types);
}

Function LowerOp(const Op& op, const Attrs& attrs, const std::vector<Type>& param_types,
                 const Type& ret_type) {
  Function func;
  std::vector<Var> params;
  for (int i = 0, n = param_types.size(); i < n; ++i) {
    auto var = raf::ir::MakeVar("", param_types[i]);
    var->checked_type_ = param_types[i];
    params.push_back(var);
  }
  func = Function(params, Call(op, {params.begin(), params.end()}, attrs), ret_type, {});
  func->body->checked_type_ = ret_type;
  func->checked_type_ = FuncType(param_types, ret_type, {}, {});
  return func;
}

void SetArgs(std::vector<DLTensor>* i, std::vector<DLTensor>* o, std::vector<TVMValue>* values,
             std::vector<int>* codes) {
  int arity = i->size() + o->size();
  values->resize(arity);
  codes->resize(arity);
  TVMArgsSetter setter(values->data(), codes->data());
  int cnt = 0;
  for (DLTensor& dlt : *i) {
    setter(cnt++, &dlt);
  }
  for (DLTensor& dlt : *o) {
    setter(cnt++, &dlt);
  }
}

void TVMOpEnv::Execute(const op::CallValues& call) {
  std::vector<TVMValue> values;
  std::vector<int> codes;
  SetArgs(&inputs, &outputs, &values, &codes);
  TVMArgs targs(values.data(), codes.data(), values.size());
  TVMRetValue rv;
  f.CallPacked(targs, &rv);
  if (call->out->IsInstance<TensorValueObj>()) {
    DLTensor* dlt = Downcast<value::TensorValue>(call->out);
    dlt->data = outputs[0].data;
  } else if (const auto* tv = call->out.as<value::TupleValueObj>()) {
    int i = 0;
    for (const auto& v : tv->fields) {
      DLTensor* dlt = Downcast<value::TensorValue>(v);
      dlt->data = outputs[i++].data;
    }
  } else {
    LOG(FATAL) << "InternalError: internal error.";
    throw;
  }
}

void TVMOpEnv::Execute(const std::vector<Value>& inputs, Value output) {
  this->inputs.clear();
  this->outputs.clear();
  for (auto val : inputs) {
    GetDLTensor(val, &this->inputs);
  }
  GetDLTensor(output, &this->outputs);
  std::vector<TVMValue> values;
  std::vector<int> codes;
  SetArgs(&this->inputs, &this->outputs, &values, &codes);
  TVMArgs targs(values.data(), codes.data(), values.size());
  TVMRetValue rv;

  // Skip the execution if we are in the task extraction mode since
  // we do not care about the correctness.
  if (AllowJitFailure()) {
    return;
  }

  f.CallPacked(targs, &rv);
}

PackedMetricMap DumpTVMCacheMetric(const std::string& cache_name) {
  static std::unordered_map<std::string, MetaCacheMetric*> name_to_cache = {
      {"tvm_cpu", &CacheBuildCpu},
      {"tvm_cuda", &CacheBuildCuda},
      {"tvm_lower", &CacheLoweredFunc},
  };

  PackedMetricMap ret;
  if (name_to_cache.count(cache_name) == 0) {
    LOG(WARNING) << "Cannot find cache " << cache_name << " for dumping metric";
    return ret;
  }

  auto metrics = name_to_cache[cache_name]->GetMetric();
  for (const auto& it : metrics) {
    ret.Set(it.first, it.second);
  }
  return ret;
}

static void CreateDir(const std::string& path) {
  if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno != EEXIST) {
      LOG(FATAL) << "Failed to create directory " << path << ": " << strerror(errno);
      throw;
    }
  }
}

static bool DeleteFile(const std::string& path) {
  if (remove(path.c_str()) == -1) {
    if (errno != ENOENT) {
      LOG(FATAL) << "Failed to delete file " << path << ": " << strerror(errno);
      throw;
    }
    return false;
  }
  return true;
}

static std::vector<char> ReadAllBytes(char const* filename)
{
    std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    std::ifstream::pos_type pos = ifs.tellg();
    if (pos == 0) {
        return std::vector<char>{};
    }
    std::vector<char> result(pos);
    ifs.seekg(0, std::ios::beg);
    ifs.read(&result[0], pos);
    return result;
}

void DumpTVMCache(const std::string& cache_dir, const std::string& cache_name, MetaPersistCache<TVMModuleCacheEntry>* cache) {
    std::string cache_path = cache_dir + "/" + cache_name + ".cache";
    LOG(INFO) << "Dumping TVM cache " << cache_name << " to " << cache_path;
    std::ofstream cache_file(cache_path, std::ios::binary);
    // Write the size of the cache first.
    int64_t cache_size = cache->Size();
    cache_file.write(reinterpret_cast<const char*>(&cache_size), sizeof(int64_t));
    for (auto& it : (*cache)) {
      // Write key first.
      int64_t key_size = it.first.size();
      cache_file.write(reinterpret_cast<const char*>(&key_size), sizeof(int64_t));
      cache_file.write(it.first.c_str(), it.first.size());
      // TVM module only supports writing to file
      // so we have to read the content back to memory
      std::string tmp_dir = cache_dir + "/" + std::to_string(std::hash<std::string>{}(it.first));
      CreateDir(tmp_dir);
      it.second.Save(tmp_dir);
      std::string tmp_fn = tmp_dir + "/tvm_module.so";
      auto content = ReadAllBytes(tmp_fn.c_str());
      std::string function_name = it.second.GetFunctionName();
      // Write function name size and content.
      int64_t function_name_size = function_name.size();
      cache_file.write(reinterpret_cast<const char*>(&function_name_size), sizeof(int64_t));
      cache_file.write(function_name.c_str(), function_name.size());
      // Write content size and content.
      int64_t content_size = content.size();
      cache_file.write(reinterpret_cast<const char*>(&content_size), sizeof(int64_t));
      cache_file.write(reinterpret_cast<const char*>(content.data()), content.size());
      std::system(("rm -r " + tmp_dir).c_str());
    }
    cache_file.close();
}

void DumpRelayCache(const std::string& cache_dir, const std::string& cache_name, MetaPersistCache<RelayFuncCacheEntry>* cache) {
    std::string cache_path = cache_dir + "/" + cache_name + ".cache";
    LOG(INFO) << "Dumping Relay cache " << cache_name << " to " << cache_path;
    std::ofstream cache_file(cache_path, std::ios::binary);
    // Write the size of the cache first.
    int64_t cache_size = cache->Size();
    cache_file.write(reinterpret_cast<const char*>(&cache_size), sizeof(int64_t));
    for (auto& it : (*cache)) {
      // Write key first.
      int64_t key_size = it.first.size();
      cache_file.write(reinterpret_cast<const char*>(&key_size), sizeof(int64_t));
      cache_file.write(it.first.c_str(), it.first.size());
      std::string json_str = it.second.GetJSONString();
      // Write json size and content.
      int64_t content_size = json_str.size();
      cache_file.write(reinterpret_cast<const char*>(&content_size), sizeof(int64_t));
      cache_file.write(reinterpret_cast<const char*>(json_str.c_str()), content_size);
    }
    cache_file.close();
}

void LoadTVMCache(const std::string& cache_dir, const std::string& cache_name, MetaPersistCache<TVMModuleCacheEntry>* cache, const std::string& tmpfile_prefix) {
  static auto f_load = registry::GetPackedFunc("raf._tvm_op.utils.load_module");

  std::string cache_path = cache_dir + "/" + cache_name + ".cache";
  LOG(INFO) << "Loading TVM cache " << cache_name << " from " << cache_path;
  std::ifstream cache_file(cache_path, std::ios::binary);
  int64_t size;
  // Read the size of the cache first.
  cache_file.read(reinterpret_cast<char*>(&size), sizeof(int64_t));
  for (size_t i=0; i<size; i++) {
    // Read key first.
    int64_t key_size;
    cache_file.read(reinterpret_cast<char*>(&key_size), sizeof(int64_t));
    std::string key(key_size, ' ');
    cache_file.read(&key[0], key_size);
    // Read func name
    int64_t function_name_size;
    cache_file.read(reinterpret_cast<char*>(&function_name_size), sizeof(int64_t));
    std::string function_name(function_name_size, ' ');
    cache_file.read(&function_name[0], function_name_size);
    // Read value. TVM module only supports loading from a file
    // so we have to write the content to a temporary file first.
    std::string tmp_fn = cache_dir + "/" + tmpfile_prefix + std::to_string(std::hash<std::string>{}(key)) + ".so";
    int64_t value_size;
    cache_file.read(reinterpret_cast<char*>(&value_size), sizeof(int64_t));
    std::vector<char> value(value_size, ' ');
    cache_file.read(&value[0], value_size);
    std::ofstream tmp_file(tmp_fn, std::ios::binary);
    tmp_file.write(&value[0], value_size);
    tmp_file.close();
    tvm::runtime::Module mod = f_load(tmp_fn);
    DeleteFile(tmp_fn);
    if (cache->Get(key) == nullptr) {
      cache->Set(key, TVMModuleCacheEntry(mod, function_name));
    }
  }
}

void LoadRelayCache(const std::string& cache_dir, const std::string& cache_name, MetaPersistCache<RelayFuncCacheEntry>* cache) {
  std::string cache_path = cache_dir + "/" + cache_name + ".cache";
  LOG(INFO) << "Loading Relay cache " << cache_name << " from " << cache_path;
  std::ifstream cache_file(cache_path, std::ios::binary);
  int64_t size;
  // Read the size of the cache first.
  cache_file.read(reinterpret_cast<char*>(&size), sizeof(int64_t));
  for (size_t i=0; i<size; i++) {
    // Read key first.
    int64_t key_size;
    cache_file.read(reinterpret_cast<char*>(&key_size), sizeof(int64_t));
    std::string key(key_size, ' ');
    cache_file.read(&key[0], key_size);
    // Read json string.
    int64_t json_size;
    cache_file.read(reinterpret_cast<char*>(&json_size), sizeof(int64_t));
    std::string func_json(json_size, ' ');
    cache_file.read(&func_json[0], json_size);
    auto func = Downcast<Function>(tvm::LoadJSON(func_json));
    if (cache->Get(key) == nullptr) {
      cache->Set(key, RelayFuncCacheEntry(func));
    }
  }
}

void DumpAllTVMCaches(const std::string& cache_dir) {
  CreateDir(cache_dir);
  LOG(INFO) << "Dumping TVM cache to " << cache_dir;
  DumpTVMCache(cache_dir, "tvm_cpu", &CacheBuildCpu);
  DumpTVMCache(cache_dir, "tvm_cuda", &CacheBuildCuda);
  DumpRelayCache(cache_dir, "tvm_lower", &CacheLoweredFunc);
}

void LoadAllTVMCaches(const std::string& cache_dir, const std::string& tmpfile_prefix) {
  LoadTVMCache(cache_dir, "tvm_cpu", &CacheBuildCpu, tmpfile_prefix);
  LoadTVMCache(cache_dir, "tvm_cuda", &CacheBuildCuda, tmpfile_prefix);
  LoadRelayCache(cache_dir, "tvm_lower", &CacheLoweredFunc);
}

RAF_REGISTER_GLOBAL("raf.cache.DumpTVMCacheMetric").set_body_typed(DumpTVMCacheMetric);
RAF_REGISTER_GLOBAL("raf.cache.DumpAllTVMCaches").set_body_typed(DumpAllTVMCaches);
RAF_REGISTER_GLOBAL("raf.cache.LoadAllTVMCaches").set_body_typed(LoadAllTVMCaches);

RAF_REGISTER_DIALECT("tvm").set_enable(DevType::kCPU()).set_enable(DevType::kCUDA());
TVM_REGISTER_PASS_CONFIG_OPTION("raf.tvm.allow_jit_failure", tvm::Bool);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
