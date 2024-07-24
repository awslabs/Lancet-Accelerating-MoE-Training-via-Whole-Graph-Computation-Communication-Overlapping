/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_pool.h
 * \brief Memory pool API
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "./device.h"
#include "raf/device_api.h"

namespace raf {
namespace memory_pool {

using device_api::DeviceAPI;

class MemoryPool;

/*!
 * \brief A wrapper for a chunk of memory, which may have shared reference to the pool so that they
 * are freed correctly. Interaction between memory pool manager also happens here.
 *
 * This wrapper is the base wrapper for memory.
 *
 * \sa Memory
 */
class Memory {
 public:
  virtual ~Memory() = default;

 public:
  static int64_t GetAllocBytes(const Device& dev, int64_t nbytes);

  static std::shared_ptr<Memory> Alloc(const Device& dev, int64_t nbytes,
                                       int64_t alignment = kDefaultMemoryAlignment);

  static std::shared_ptr<Memory> AllocAsync(const Device& dev, int64_t nbytes, void* stream,
                                            int64_t alignment = kDefaultMemoryAlignment);

  static std::vector<std::shared_ptr<Memory> > AllocBatch(
      const Device& dev, const std::vector<int64_t>& nbytes,
      int64_t alignment = kDefaultMemoryAlignment);

  static std::pair<float, float> GetPoolSize(const Device& dev);

  // means "no longer considered as allocator when asking for new memory."
  static void RemovePool(const Device& dev);

  static MemoryPool* ResetPool(const Device& dev);

  static MemoryPool* GetPool(const Device& dev);

  static MemoryPool* InitPool(const Device& dev, const std::string& name);

  virtual void SetStream(void* stream) {
    // do nothing by default
  }

 public:
  /*! \brief The pointer to the allocated chunk of memory. */
  void* data = nullptr;
  /*! \brief The context of the allocated chunk of memory. */
  Device device{};
};

class NonOwnedMemory final : public Memory {
 public:
  explicit NonOwnedMemory(void* data, const Device& dev, std::shared_ptr<DeviceAPI> api);
  ~NonOwnedMemory();

 public:
  std::shared_ptr<DeviceAPI> api;
};

class NonOwnedAsyncMemory final : public Memory {
 public:
  explicit NonOwnedAsyncMemory(void* data, void* stream, const Device& dev,
                               std::shared_ptr<DeviceAPI> api);
  ~NonOwnedAsyncMemory();

  void SetStream(void* stream) override;

 public:
  std::shared_ptr<DeviceAPI> api;
  void* stream;
};

/*!
 * \brief A base class for memory pool.
 * Only interface for implementing new allocation strategy, no static interface is included.
 */
class MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  /*!
   * \brief Get the name of this memory pool.
   * \return The memory pool name.
   */
  virtual std::string GetName() = 0;

  /*!
   * \brief Calculate the actual bytes to be allocated. This may be different as the requested
   * size due to alignment or page unit.
   * \param nbytes The requested bytes to be allocated.
   */
  virtual int64_t GetAllocBytes(int64_t nbytes) = 0;

  /*!
   * \brief A helper function to change bytes to mega-bytes.
   *
   * \param nbytes The number in bytes.
   *
   * \return The number in MBs.
   */
  inline float BytesToMegaBytes(float nbytes) {
    return nbytes / 1048576.0;
  }

  /*!
   * \brief Allocate a chunk of memory with given size and alignment.
   *
   * \param nbytes The size of the memory chunk to allocate.
   * \param alignment The alignment of the memory chunk to allocate.
   *
   * \return A shared pointer to Memory object which holds the memory chunk.
   */
  virtual std::shared_ptr<Memory> Alloc(int64_t nbytes,
                                        int64_t alignment = kDefaultMemoryAlignment) = 0;

  virtual std::shared_ptr<Memory> AllocAsync(int64_t nbytes, void* stream,
                                             int64_t alignment = kDefaultMemoryAlignment) = 0;

  /*!
   * \brief Allocate a bacth of memory chunks with given sizes and alignments.
   *
   * \param nbytes The sizes of the memory chunks to allocate.
   * \param alignment The alignments of the memory chunks to allocate.
   *
   * \return The shared pointers to Memory object which hold the memory chunks.
   */
  virtual std::vector<std::shared_ptr<Memory> > AllocBatch(
      const std::vector<int64_t>& nbytes, int64_t alignment = kDefaultMemoryAlignment) = 0;

  /*!
   * \brief Get the current pool size in MBs.
   *
   * \return A pair of the total size of (used chunks, pool).
   */
  virtual std::pair<float, float> GetPoolSize() = 0;
};

namespace no_pool {

class NoPool final : public MemoryPool {
 public:
  class NonOwnedReservedAsyncMemory final : public Memory {
  public:
    explicit NonOwnedReservedAsyncMemory(void* data, void* stream, const Device& dev,
                                NoPool* pool, int64_t nbytes);
    ~NonOwnedReservedAsyncMemory();
  public:
    NoPool* pool;
    void* stream;
    int64_t nbytes;
  };

  struct MemoryFreeCallbackArgs {
    MemoryFreeCallbackArgs(NoPool* pool, void* data, void* stream, int64_t nbytes):
      pool(pool), data(data), stream(stream), nbytes(nbytes) {}
    NoPool* pool;
    void* data;
    void* stream;
    int64_t nbytes;
  };

  explicit NoPool(Device dev);
  std::string GetName();
  int64_t GetAllocBytes(int64_t nbytes) override;
  std::shared_ptr<Memory> Alloc(int64_t nbytes, int64_t alignment) override;

  std::shared_ptr<Memory> AllocAsync(int64_t nbytes, void* stream,
                                     int64_t alignment = kDefaultMemoryAlignment) override;

  void FreeReservedMemoryAsync(int64_t nbytes, void* data, void* stream);

  std::vector<std::shared_ptr<Memory>> AllocBatch(const std::vector<int64_t>& nbytes,
                                                  int64_t alignment) override;

  std::pair<float, float> GetPoolSize() override;

 public:
  static void* make(const Device& dev) {
    return new NoPool(dev);
  }

  Device device;
  std::shared_ptr<DeviceAPI> api;
  std::unordered_map<int64_t, std::list<std::shared_ptr<Memory>>> reserved_memory;
  std::unordered_map<void*,
    std::unordered_map<int64_t,
      std::list<std::shared_ptr<Memory>>>> available_reserved_memory_per_stream;
  std::list<std::shared_ptr<MemoryFreeCallbackArgs>> callback_args;
  std::mutex reserved_memory_mutex;
  std::mutex callback_args_mutex;
};
} // namespace no_pool


}  // namespace memory_pool
}  // namespace raf
