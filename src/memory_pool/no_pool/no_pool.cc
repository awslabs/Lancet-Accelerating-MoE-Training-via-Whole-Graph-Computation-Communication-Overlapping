/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/memory_pool/no_pool/no_pool.cc
 * \brief No memory pool
 */
#include <atomic>
#include <mutex>
#include "raf/device.h"
#include "raf/device_api.h"
#include "raf/memory_pool.h"
#include "raf/registry.h"

namespace raf {
namespace memory_pool {

using device_api::DeviceAPI;

NonOwnedMemory::NonOwnedMemory(void* data, const Device& dev, std::shared_ptr<DeviceAPI> api) {
  this->data = data;
  this->device = dev;
  this->api = std::move(api);
}

NonOwnedMemory::~NonOwnedMemory() {
  if (data != nullptr) {
    api->FreeMemory(data);
  }
}

NonOwnedAsyncMemory::NonOwnedAsyncMemory(void* data, void* stream, const Device& dev,
                              std::shared_ptr<DeviceAPI> api) {
  this->data = data;
  this->stream = stream;
  this->device = dev;
  this->api = std::move(api);
}

NonOwnedAsyncMemory::~NonOwnedAsyncMemory() {
  if (data != nullptr) {
    api->FreeMemoryAsync(data, stream);
  }
}

void NonOwnedAsyncMemory::SetStream(void* stream) {
  this->stream = stream;
}

namespace no_pool {

static int64_t kReservedMemoryThreshold = 64 * 1024 * 1024;

NoPool::NoPool(Device dev) {
  this->device = dev;
  this->api = DeviceAPI::Get(dev.device_type());

  if (dev.device_type() == DevType::kCUDA()) {
    this->api->SetDevice(dev.device_id());
  }
  if (const char* val = getenv("RAF_RESERVED_MEMORY_THRESHOLD")) {
      kReservedMemoryThreshold = atol(val);
    }
}

std::string NoPool::GetName() {
  return "no_pool";
}

int64_t NoPool::GetAllocBytes(int64_t nbytes) {
  return nbytes;
}

std::shared_ptr<Memory> NoPool::Alloc(int64_t nbytes, int64_t alignment) {
  CHECK_GE(nbytes, 0);
  void* data = nullptr;
  if (nbytes > 0) {
    data = api->AllocMemory(nbytes, alignment);
  }
  return std::make_shared<NonOwnedMemory>(data, device, api);
}

std::shared_ptr<Memory> NoPool::AllocAsync(int64_t nbytes, void* stream,
                                           int64_t alignment) {
  CHECK_GE(nbytes, 0);
  void* data = nullptr;
  if (nbytes > 0) {
    if(nbytes >= kReservedMemoryThreshold) {
      // LOG(INFO) << "Need to alloc reserved memory: " << BytesToMegaBytes(nbytes) << " MB.";
      std::lock_guard<std::mutex> lock(reserved_memory_mutex);
      if(available_reserved_memory_per_stream.count(stream)) {
        auto& available_memory_at_stream = available_reserved_memory_per_stream.at(stream);
        if(available_memory_at_stream.count(nbytes)) {
          auto& available_memory_sized_nbytes = available_memory_at_stream.at(nbytes);
          if(!available_memory_sized_nbytes.empty()) {
            // directly available
            // LOG(INFO) << "Found available reserved memory: " << BytesToMegaBytes(nbytes) << " MB. " << available_reserved_memory_per_stream[stream][nbytes].size() << " blocks available.";
            auto memory = available_memory_sized_nbytes.back();
            available_memory_sized_nbytes.pop_back();
            // LOG(INFO) << "After distributing the block: " << available_reserved_memory_per_stream[stream][nbytes].size() << " blocks available.";
            return memory;
          }
        }
      }
      // need allocation
      // LOG(INFO) << "Allocated new reserved memory: " << BytesToMegaBytes(nbytes) << " MB.";
      data = api->AllocMemoryAsync(nbytes, stream, alignment);
      auto async_memory = std::make_shared<NonOwnedAsyncMemory>(data, stream, device, api);
      reserved_memory[nbytes].push_back(std::move(async_memory));
      auto res_memory = std::make_shared<NonOwnedReservedAsyncMemory>(data, stream, device, this, nbytes);
      return res_memory;
    } else {
      data = api->AllocMemoryAsync(nbytes, stream, alignment);
    }
  }
  return std::make_shared<NonOwnedAsyncMemory>(data, stream, device, api);
}

void NoPool::FreeReservedMemoryAsync(int64_t nbytes, void* data, void* stream) {
  // LOG(INFO) << "Inserting free reserved memory callback: " << BytesToMegaBytes(nbytes) << " MB.";
  // auto callback_arg = std::make_shared<MemoryFreeCallbackArgs>(this, data, stream, nbytes);
  // LOG(INFO) << "Inserting callback args at address: " << (void*)callback_arg.get() << " .";
  // {
  //   std::lock_guard<std::mutex> lock(callback_args_mutex);
  //   callback_args.push_back(callback_arg);
  // }
  // api->StreamRegisterCallback(FreeMemoryAsyncCallback, stream, static_cast<void*>(callback_arg.get()));
  auto memory = std::make_shared<NonOwnedReservedAsyncMemory>(data, stream, device, this, nbytes);
  {
    std::lock_guard<std::mutex> lock(reserved_memory_mutex);
    available_reserved_memory_per_stream[stream][nbytes].push_back(std::move(memory));
    // LOG(INFO) << "Added back available memory: " << BytesToMegaBytes(nbytes) << " MB. Afterwards, " << available_reserved_memory_per_stream[stream][nbytes].size() << " blocks available.";
  }
}

std::vector<std::shared_ptr<Memory>> NoPool::AllocBatch(const std::vector<int64_t>& nbytes,
                                                  int64_t alignment) {
  std::vector<std::shared_ptr<Memory>> ret;
  ret.reserve(nbytes.size());
  for (int64_t bytes : nbytes) {
    ret.emplace_back(Alloc(bytes, alignment));
  }
  return ret;
}

std::pair<float, float> NoPool::GetPoolSize() {
  auto ret = api->GetPoolSize();
  return {BytesToMegaBytes(ret.first), BytesToMegaBytes(ret.second)};
}

NoPool::NonOwnedReservedAsyncMemory::NonOwnedReservedAsyncMemory(void* data, void* stream, const Device& dev,
                                NoPool* pool, int64_t nbytes) {
  this->data = data;
  this->stream = stream;
  this->device = dev;
  this->pool = std::move(pool);
  this->nbytes = nbytes;
}

NoPool::NonOwnedReservedAsyncMemory::~NonOwnedReservedAsyncMemory() {
  if (data != nullptr) {
    pool->FreeReservedMemoryAsync(nbytes, data, stream);
  }
}

RAF_REGISTER_GLOBAL("raf.memory_pool._make.no_pool").set_body_typed([](const Device& dev) {
  return NoPool::make(dev);
});

}  // namespace no_pool
}  // namespace memory_pool
}  // namespace raf
