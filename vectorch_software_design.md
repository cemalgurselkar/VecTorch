# VecTorch --- V1 Software Architecture & Design

> **Amaç:** Python içinde çalışan, single-node, embedded, persistent bir
> vector database / ANN search engine geliştirmek.\
> V1 tasarımında öncelik: **temiz sorumluluk ayrımı, değiştirilebilir
> index/kernel katmanları ve performans deneylerine uygun mimari.**

------------------------------------------------------------------------

## 1. İlişki Notasyonu

Bu dokümandaki diyagramlarda:

``` text
A ──owns──> B         A, B'nin yaşam döngüsünden sorumludur / B'yi içerir.
A ──uses──> B         A, B'yi kullanır fakat B değildir.
A ─implements──> I    A, I interface/abstract class sözleşmesini uygular.
```

**Inheritance yalnızca açıkça `implements` yazılan yerlerde
düşünülmelidir.**

------------------------------------------------------------------------

# 2. Nihai Üst Seviye Mimari

``` text
┌───────────────────────────────────────────┐
│                  VecTorch                 │
├───────────────────────────────────────────┤
│ - manager: CollectionManager              │
│ - root_path: Path                         │
├───────────────────────────────────────────┤
│ + __init__(path)                          │
│ + create_collection(config) -> Collection │
│ + get_collection(name) -> Collection      │
│ + drop_collection(name)                   │
│ + list_collections() -> list[str]         │
│ + close()                                 │
└────────────────────┬──────────────────────┘
                     │ owns
                     ▼
┌───────────────────────────────────────────┐
│            CollectionManager              │
├───────────────────────────────────────────┤
│ - collections: dict[str, Collection]      │
│ - root_path: Path                         │
├───────────────────────────────────────────┤
│ + create(config) -> Collection            │
│ + get(name) -> Collection                 │
│ + drop(name)                              │
│ + list() -> list[str]                     │
│ + load_all()                              │
└────────────────────┬──────────────────────┘
                     │ owns 0..N
                     ▼
┌───────────────────────────────────────────┐
│                 Collection                │
├───────────────────────────────────────────┤
│ - config: CollectionConfig                │
│ - storage: StorageEngine                  │
│ - index: Index                            │
│ - persistence: PersistenceManager         │
├───────────────────────────────────────────┤
│ + add(ids, vectors, metadata=None)        │
│ + search(query, k, filter=None)           │
│ + get(id) -> Record | None                │
│ + delete(ids)                             │
│ + count() -> int                          │
│ + save()                                 │
│ + close()                                │
└──────────┬─────────────┬──────────────────┘
           │ owns        │ uses
           ▼             ▼
    StorageEngine       Index
           │             │
           │             ├── uses VectorStore
           │             └── uses DistanceKernel
           │
           └── owns VectorStore / IDMap /
                    MetadataStore / DeleteBitmap
```

------------------------------------------------------------------------

# 3. Public API Katmanı

## 3.1 `VecTorch`

**Görevi:** Kullanıcının sisteme giriş noktasıdır.

Bu sınıf vector arama algoritması veya storage detayları bilmez.

``` text
┌──────────────────────────────────────────────┐
│                  VecTorch                    │
├──────────────────────────────────────────────┤
│ - root_path: Path                            │
│ - manager: CollectionManager                 │
├──────────────────────────────────────────────┤
│ + __init__(path)                             │
│ + create_collection(...) -> Collection       │
│ + get_collection(name) -> Collection         │
│ + drop_collection(name)                      │
│ + list_collections() -> list[str]            │
│ + close()                                    │
└──────────────────────────────────────────────┘
```

Örnek public API:

``` python
from vectorch import VecTorch

db = VecTorch("./data")

docs = db.create_collection(
    name="documents",
    dimension=768,
    metric="cosine",
    index="hnsw",
)

docs.add(ids, vectors, metadata)
results = docs.search(query, k=10)

db.close()
```

### Neden `vectorch.py`?

`VecTorch` public entry-point/facade sınıfıdır. Bu nedenle:

``` text
src/vectorch/vectorch.py
```

dosyasında bulunması `database.py` isminden daha açıktır.

`src/vectorch/__init__.py`:

``` python
from .vectorch import VecTorch

__all__ = ["VecTorch"]
```

Böylece kullanıcı:

``` python
from vectorch import VecTorch
```

yazar.

------------------------------------------------------------------------

# 4. Collection Yönetimi

## 4.1 `CollectionManager`

**Görevi:** Collection lifecycle yönetimi.

Vector eklemek veya aramak onun görevi değildir.

``` text
┌─────────────────────────────────────────────┐
│             CollectionManager               │
├─────────────────────────────────────────────┤
│ - collections: dict[str, Collection]        │
│ - root_path: Path                           │
├─────────────────────────────────────────────┤
│ + create(config) -> Collection              │
│ + get(name) -> Collection                   │
│ + drop(name)                                │
│ + list() -> list[str]                       │
│ + load_all()                                │
└─────────────────────────────────────────────┘
```

Sorumluluk:

``` text
Collection oluştur
Collection bul
Collection sil
Diskteki collection'ları yükle
```

**Yapmaması gerekenler:**

``` text
add()
search()
delete(vector)
distance hesaplama
HNSW traversal
```

### Factory konusu

V1'de ayrı `CollectionFactory` yok.

`CollectionManager.create()` construction işini üstlenebilir.

İleride construction karmaşıklaşırsa:

``` text
CollectionManager
       │
       ▼
CollectionFactory
       │
       ├── StorageEngine
       ├── Index
       ├── DistanceKernel
       └── PersistenceManager
```

şeklinde ayrılabilir.

------------------------------------------------------------------------

# 5. Collection

`Collection`, kullanıcı açısından asıl vector database çalışma alanıdır.

``` text
┌──────────────────────────────────────────────────┐
│                    Collection                    │
├──────────────────────────────────────────────────┤
│ - config: CollectionConfig                       │
│ - storage: StorageEngine                         │
│ - index: Index                                   │
│ - persistence: PersistenceManager                │
├──────────────────────────────────────────────────┤
│ + add(ids, vectors, metadata=None)               │
│ + search(query, k, filter=None)                  │
│ + get(id) -> Record | None                       │
│ + delete(ids)                                    │
│ + count() -> int                                 │
│ + save()                                         │
│ + close()                                        │
│                                                  │
│ V1.1:                                            │
│ + search_batch(queries, k)                       │
└──────────────────────────────────────────────────┘
```

### Collection'ın görevi

Collection **orchestrator** gibi davranır.

Örneğin `add()`:

``` text
Collection
   │
   ├── input validation
   ├── StorageEngine.add()
   ├── Index.add()
   └── persistence state güncelle
```

Collection distance hesabını kendi yazmaz.

Collection HNSW traversal yapmaz.

Collection vectorları Python listesinde saklamaz.

------------------------------------------------------------------------

# 6. Storage Katmanı

## 6.1 `StorageEngine`

Storage alt bileşenlerini tek noktada toplar.

``` text
┌───────────────────────────────────────────────┐
│               StorageEngine                   │
├───────────────────────────────────────────────┤
│ - vectors: VectorStore                        │
│ - ids: IDMap                                  │
│ - metadata: MetadataStore                     │
│ - deleted: DeleteBitmap                       │
├───────────────────────────────────────────────┤
│ + add(ext_id, vector, metadata) -> internal_id│
│ + get(internal_id)                            │
│ + delete(internal_id)                         │
│ + is_deleted(internal_id) -> bool             │
│ + count() -> int                              │
└───────────────────────────────────────────────┘
```

StorageEngine'ın görevi **veriyi yönetmek**, arama algoritmasını
çalıştırmak değildir.

------------------------------------------------------------------------

## 6.2 `VectorStore`

Vector hot-data burada bulunur.

``` text
┌──────────────────────────────────────────┐
│               VectorStore                │
├──────────────────────────────────────────┤
│ - data: np.ndarray                       │
│ - dimension: int                         │
│ - size: int                              │
│ - capacity: int                          │
├──────────────────────────────────────────┤
│ + append(vector) -> internal_id          │
│ + get(internal_id) -> ndarray            │
│ + get_many(ids) -> ndarray               │
│ + reserve(capacity)                      │
│ + __len__() -> int                       │
└──────────────────────────────────────────┘
```

Başlangıç veri düzeni:

``` text
shape = (capacity, dimension)
dtype = float32
```

Amaç:

-   contiguous memory
-   cache-friendly traversal
-   NumPy/Numba/native kodla kolay paylaşım
-   ileride alignment / mmap / zero-copy deneyleri

------------------------------------------------------------------------

## 6.3 `IDMap`

Kullanıcının ID'si ile engine'in compact integer ID'sini ayırır.

``` text
┌──────────────────────────────────────────┐
│                  IDMap                   │
├──────────────────────────────────────────┤
│ - external_to_internal                   │
│ - internal_to_external                   │
├──────────────────────────────────────────┤
│ + add(external_id) -> internal_id        │
│ + internal(external_id) -> internal_id   │
│ + external(internal_id) -> external_id   │
│ + remove(external_id)                    │
│ + contains(external_id) -> bool          │
└──────────────────────────────────────────┘
```

Index yapıları mümkün olduğunca compact `internal_id` kullanır.

------------------------------------------------------------------------

## 6.4 `MetadataStore`

``` text
┌──────────────────────────────────────────┐
│             MetadataStore                │
├──────────────────────────────────────────┤
│ - data                                   │
├──────────────────────────────────────────┤
│ + set(internal_id, metadata)             │
│ + get(internal_id)                       │
│ + get_many(ids)                          │
│ + delete(internal_id)                    │
└──────────────────────────────────────────┘
```

Metadata vector memory'den ayrı tutulur.

Sebep: Python dictionary/string/object gibi verileri search hot-path'ten
uzak tutmak.

------------------------------------------------------------------------

## 6.5 `DeleteBitmap`

V1'de fiziksel compaction yerine logical deletion kullanılabilir.

``` text
┌──────────────────────────────────────────┐
│              DeleteBitmap                │
├──────────────────────────────────────────┤
│ - deleted                                │
├──────────────────────────────────────────┤
│ + mark(internal_id)                      │
│ + unmark(internal_id)                    │
│ + contains(internal_id) -> bool          │
│ + clear()                                │
└──────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 7. Index Katmanı

## 7.1 `Index` Interface

``` text
┌────────────────────────────────────────────┐
│            <<interface>> Index             │
├────────────────────────────────────────────┤
│ + add(internal_id)                         │
│ + remove(internal_id)                      │
│ + search(query, k) -> CandidateIDs         │
│ + build()                                  │
│ + save(path)                               │
│ + load(path)                               │
└────────────────────────────────────────────┘
                  ▲
                  │ implements
        ┌─────────┼──────────┐
        │         │          │
        ▼         ▼          ▼
   FlatIndex   HNSWIndex   IVFIndex
```

Bu katmanda **Strategy Pattern** kullanıyoruz.

Collection concrete index türünü bilmek zorunda değildir.

------------------------------------------------------------------------

## 7.2 `FlatIndex`

Exact baseline.

``` text
┌──────────────────────────────────────────┐
│                FlatIndex                 │
├──────────────────────────────────────────┤
│ - vectors: VectorAccessor                │
│ - kernel: DistanceKernel                 │
├──────────────────────────────────────────┤
│ + add(internal_id)                       │
│ + remove(internal_id)                    │
│ + search(query, k)                       │
│ + build()                                │
│ + save(path)                             │
│ + load(path)                             │
└──────────────────────────────────────────┘
```

Amaçları:

1.  çalışan ilk search implementation
2.  exact search
3.  ANN algoritmaları için ground truth
4.  Recall@K benchmark referansı

------------------------------------------------------------------------

## 7.3 `HNSWIndex`

``` text
┌──────────────────────────────────────────┐
│                HNSWIndex                 │
├──────────────────────────────────────────┤
│ - graph                                  │
│ - entry_point                            │
│ - max_level                              │
│ - M                                      │
│ - ef_construction                        │
│ - ef_search                              │
│ - vectors: VectorAccessor                │
│ - kernel: DistanceKernel                 │
├──────────────────────────────────────────┤
│ + add(internal_id)                       │
│ + remove(internal_id)                    │
│ + search(query, k)                       │
│ + build()                                │
│ + save(path)                             │
│ + load(path)                             │
│                                          │
│ internal helpers:                        │
│ - _search_layer(...)                     │
│ - _select_neighbors(...)                 │
│ - _random_level()                        │
│ - _connect(...)                          │
└──────────────────────────────────────────┘
```

HNSW'nin kendi algoritmik state'i burada kalır.

------------------------------------------------------------------------

## 7.4 `IVFIndex`

``` text
┌──────────────────────────────────────────┐
│                 IVFIndex                 │
├──────────────────────────────────────────┤
│ - centroids                              │
│ - inverted_lists                         │
│ - nlist                                  │
│ - nprobe                                 │
│ - vectors: VectorAccessor                │
│ - kernel: DistanceKernel                 │
├──────────────────────────────────────────┤
│ + train()                                │
│ + add(internal_id)                       │
│ + remove(internal_id)                    │
│ + search(query, k)                       │
│ + build()                                │
│ + save(path)                             │
│ + load(path)                             │
└──────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 8. Vector Access Sınırı

Index'in bütün `StorageEngine`'e bağımlı olması kötü olur.

Index'in ihtiyacı esas olarak vector erişimidir.

Bu nedenle ilerleyen aşamada dar bir interface kullanılabilir:

``` text
┌──────────────────────────────────────────┐
│       <<interface>> VectorAccessor       │
├──────────────────────────────────────────┤
│ + get(id) -> vector                      │
│ + get_many(ids) -> vectors               │
│ + count() -> int                         │
└──────────────────────────────────────────┘
                 ▲
                 │ implements
                 │
             VectorStore
```

Böylece:

``` text
HNSWIndex ──uses──> VectorAccessor
```

ama:

``` text
HNSWIndex ─X─> MetadataStore
HNSWIndex ─X─> IDMap
HNSWIndex ─X─> PersistenceManager
```

Bu dependency sınırı önemlidir.

------------------------------------------------------------------------

# 9. Distance / HPC Kernel Katmanı

Index algoritması ile distance hesaplamasının implementation'ını
ayırıyoruz.

``` text
┌───────────────────────────────────────────┐
│      <<interface>> DistanceKernel         │
├───────────────────────────────────────────┤
│ + distance(query, vectors, metric)        │
│ + l2(query, vectors)                      │
│ + dot(query, vectors)                     │
│ + cosine(query, vectors)                  │
└────────────────────▲──────────────────────┘
                     │ implements
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
     NumPyKernel  NumbaKernel  NativeSIMDKernel
```

## `NumPyKernel`

``` text
┌──────────────────────────────────────────┐
│              NumPyKernel                 │
├──────────────────────────────────────────┤
│ + l2(query, vectors)                     │
│ + dot(query, vectors)                    │
│ + cosine(query, vectors)                 │
└──────────────────────────────────────────┘
```

İlk correctness/performance baseline.

## `NumbaKernel`

``` text
┌──────────────────────────────────────────┐
│              NumbaKernel                 │
├──────────────────────────────────────────┤
│ + l2(query, vectors)                     │
│ + dot(query, vectors)                    │
│ + cosine(query, vectors)                 │
└──────────────────────────────────────────┘
```

JIT, loop optimization ve parallel execution deneyleri burada yapılır.

## `NativeSIMDKernel`

Daha sonra:

``` text
┌──────────────────────────────────────────┐
│           NativeSIMDKernel               │
├──────────────────────────────────────────┤
│ + l2(query, vectors)                     │
│ + dot(query, vectors)                    │
│ + cosine(query, vectors)                 │
└──────────────────────────────────────────┘

       C/C++
         │
         ├── AVX2
         ├── AVX-512
         ├── FMA
         └── native threading
```

------------------------------------------------------------------------

# 10. Projenin HPC / Systems Engineering İmzası

Amaç yalnızca HNSW veya IVF implement etmek değildir.

Aynı algoritmanın farklı execution backend'lerini karşılaştırabilmek
istiyoruz.

``` text
                   HNSW
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       NumPy      Numba    Native SIMD
                              │
                       AVX2 / AVX-512
```

Araştırılacak optimizasyon alanları:

``` text
Distance computation
    ↓
JIT
    ↓
SIMD
    ↓
memory alignment
    ↓
cache locality
    ↓
allocation reduction
    ↓
thread-local buffers
    ↓
multithreading
    ↓
CPU affinity
    ↓
mmap / zero-copy
    ↓
quantization
```

Her optimizasyon benchmark ile doğrulanmalı.

Temel çalışma prensibi:

``` text
MEASURE
   ↓
PROFILE
   ↓
HYPOTHESIS
   ↓
OPTIMIZE
   ↓
MEASURE AGAIN
```

------------------------------------------------------------------------

# 11. Persistence

``` text
┌──────────────────────────────────────────┐
│          PersistenceManager              │
├──────────────────────────────────────────┤
│ - collection_path: Path                  │
├──────────────────────────────────────────┤
│ + save(storage, index)                   │
│ + load()                                 │
│ + flush()                                │
│ + close()                                │
└──────────────────────────────────────────┘
```

Disk layout:

``` text
vectorch-data/
│
├── manifest.json
│
└── collections/
    └── documents/
        ├── config.json
        ├── vectors.bin
        ├── ids.bin
        ├── metadata.bin
        ├── deleted.bin
        └── index.bin
```

`JSON` config/manifest gibi cold-path verilerde kullanılabilir.

Vector verileri JSON olarak saklanmaz.

------------------------------------------------------------------------

# 12. Add Akışı

``` text
User
 │
 │ collection.add(...)
 ▼
┌──────────────┐
│  Collection  │
└──────┬───────┘
       │ validate
       ▼
┌───────────────┐
│ StorageEngine │
└──────┬────────┘
       │
       ├── IDMap.add()
       │       ↓
       │   internal_id
       │
       ├── VectorStore.append()
       │
       └── MetadataStore.set()
       │
       ▼
┌──────────────┐
│    Index     │
│ add(int_id)  │
└──────┬───────┘
       │
       ▼
 persistence dirty
```

------------------------------------------------------------------------

# 13. Search Akışı

``` text
User
 │
 │ collection.search(query, k)
 ▼
┌──────────────┐
│  Collection  │
└──────┬───────┘
       │ validate
       ▼
┌──────────────┐
│    Index     │
└──────┬───────┘
       │
       ├── VectorAccessor
       │
       ├── DistanceKernel
       │
       └── Top-K internal IDs
              │
              ▼
        DeleteBitmap
              │
              ▼
            IDMap
     internal → external
              │
              ▼
        MetadataStore
              │
              ▼
        SearchResult[]
```

Önemli prensip:

**Metadata mümkün olduğunca Top-K belirlendikten sonra çözülür.**

Böylece Python object/string/dictionary verileri hot-path'i kirletmez.

------------------------------------------------------------------------

# 14. Delete Akışı

V1:

``` text
collection.delete(external_id)
        │
        ▼
      IDMap
 external → internal
        │
        ▼
  DeleteBitmap.mark()
        │
        ▼
    Index.remove()
```

Başlangıçta physical compaction yapmak zorunda değiliz.

Compaction daha sonraki milestone olabilir.

------------------------------------------------------------------------

# 15. Configuration / Types

## `CollectionConfig`

``` text
┌──────────────────────────────────────┐
│          CollectionConfig            │
├──────────────────────────────────────┤
│ name: str                            │
│ dimension: int                       │
│ metric: Metric                       │
│ index_type: IndexType                │
│ index_params: dict                   │
└──────────────────────────────────────┘
```

## `SearchResult`

``` text
┌──────────────────────────────────────┐
│            SearchResult              │
├──────────────────────────────────────┤
│ id                                  │
│ score: float                         │
│ metadata                             │
└──────────────────────────────────────┘
```

## Enum benzeri tipler

``` text
Metric
├── COSINE
├── L2
└── DOT

IndexType
├── FLAT
├── HNSW
└── IVF
```

String parsing public API sınırında yapılabilir; içeride typed değer
kullanmak daha güvenlidir.

------------------------------------------------------------------------

# 16. Index Seçimi

V1'de kullanıcı açıkça seçer:

``` python
db.create_collection(
    name="docs",
    dimension=768,
    metric="cosine",
    index="hnsw",
)
```

Dokümantasyon indexlerin trade-off'larını açıklar.

``` text
Flat
→ exact
→ küçük dataset
→ correctness / ground truth

HNSW
→ ANN
→ düşük latency
→ yüksek recall
→ graph nedeniyle ek RAM

IVF
→ ANN
→ partition-based search
→ nlist / nprobe tuning
```

V1'de otomatik router yok.

Daha sonra araştırılabilecek özellik:

``` python
index="auto"
```

Bu durumda dataset/workload/memory budget gibi bilgilerden index seçen
mekanizma ayrıca tasarlanabilir.

------------------------------------------------------------------------

# 17. Benchmark Architecture

Benchmark production runtime'ın parçası değildir.

``` text
┌─────────────────────────────────────────┐
│             BenchmarkRunner             │
├─────────────────────────────────────────┤
│ + benchmark_index(...)                  │
│ + benchmark_kernel(...)                 │
│ + benchmark_concurrency(...)            │
│ + compare_engines(...)                  │
└─────────────────────────────────────────┘
          │
          ├── Flat
          ├── HNSW
          ├── IVF
          ├── NumPy
          ├── Numba
          └── SIMD
```

Ölçümler:

``` text
p50 latency
p95 latency
p99 latency
p99.9 latency
QPS
Recall@K
RAM
CPU utilization
index build time
```

Flat exact search, ANN Recall@K ölçümünde ground truth olarak
kullanılabilir.

Harici karşılaştırmalar daha sonra:

``` text
VecTorch
FAISS
hnswlib
Qdrant
```

Aynı dataset, dimension, metric, hardware, thread count ve yaklaşık aynı
recall hedefi kullanılmadan performans iddiası yapılmamalı.

------------------------------------------------------------------------

# 18. Concurrency Tasarım Hedefi

İlk hedef:

``` text
SINGLE WRITER
MULTIPLE READERS
```

Üç farklı performans seviyesi ayrı düşünülmeli:

``` text
1. SIMD
   tek core içinde paralel hesap

2. Intra-query parallelism
   tek query'nin birden fazla core kullanması

3. Inter-query parallelism
   birden fazla query'nin eş zamanlı çalışması
```

İleride:

``` text
RW locks
thread-local SearchContext
immutable snapshots
copy-on-write
native threads
GIL release
CPU affinity
false sharing
NUMA
```

Bunlar V1'in ilk kodlama aşamasına eklenmemeli; benchmark ihtiyacı
ortaya çıktıkça eklenmeli.

------------------------------------------------------------------------

# 19. SearchContext --- İleri Performans Optimizasyonu

Search sırasında sürekli allocation yapmak p99 latency'yi bozabilir.

Daha sonra:

``` text
┌──────────────────────────────────────┐
│            SearchContext             │
├──────────────────────────────────────┤
│ - visited_buffer                     │
│ - candidate_buffer                   │
│ - distance_buffer                    │
│ - result_buffer                      │
├──────────────────────────────────────┤
│ + reset()                            │
└──────────────────────────────────────┘
```

Thread-local veya reusable olabilir.

V1'in ilk implementation'ında gerekli değildir.

------------------------------------------------------------------------

# 20. Nihai Dependency Graph

``` text
VecTorch
   │
   │ owns
   ▼
CollectionManager
   │
   │ owns 0..N
   ▼
Collection
   │
   ├──────── owns ────────> StorageEngine
   │                           │
   │                           ├── owns → VectorStore
   │                           ├── owns → IDMap
   │                           ├── owns → MetadataStore
   │                           └── owns → DeleteBitmap
   │
   ├──────── uses ────────> Index <<interface>>
   │                           ▲
   │                           │ implements
   │                    ┌──────┼──────┐
   │                    │      │      │
   │                  Flat   HNSW    IVF
   │                    │      │      │
   │                    └──────┼──────┘
   │                           │ uses
   │                           ├────────> VectorAccessor
   │                           │
   │                           └────────> DistanceKernel <<interface>>
   │                                          ▲
   │                                          │ implements
   │                                   ┌──────┼──────┐
   │                                   │      │      │
   │                                 NumPy  Numba   SIMD
   │
   └──────── uses ────────> PersistenceManager
```

## Kritik dependency kuralları

``` text
CollectionManager → Collection        EVET
Collection → StorageEngine            EVET
Collection → Index interface          EVET
Index → VectorAccessor                EVET
Index → DistanceKernel interface      EVET

Index → MetadataStore                 HAYIR
Index → CollectionManager             HAYIR
Index → VecTorch                      HAYIR
DistanceKernel → Index                HAYIR
StorageEngine → HNSWIndex             HAYIR
VectorStore → Collection              HAYIR
```

Bağımlılık mümkün olduğunca **üst seviyeden alt seviyeye** akar.

------------------------------------------------------------------------

# 21. Proje Klasör Yapısı

``` text
vectorch/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── vectorch/
│       │
│       ├── __init__.py
│       ├── vectorch.py
│       ├── manager.py
│       ├── collection.py
│       ├── types.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── vectors.py
│       │   ├── ids.py
│       │   ├── metadata.py
│       │   └── deleted.py
│       │
│       ├── index/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── flat.py
│       │   ├── hnsw.py
│       │   └── ivf.py
│       │
│       ├── kernels/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── numpy.py
│       │   ├── numba.py
│       │   └── native.py
│       │
│       └── persistence/
│           ├── __init__.py
│           └── manager.py
│
├── native/
│   └── ...                 # native SIMD aşamasında
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── benchmarks/
│
└── experiments/
```

Her dosyayı ilk gün oluşturmak zorunda değilsin.

Bu yapı **hedef mimaridir**, başlangıç commit'i değildir.

------------------------------------------------------------------------

# 22. Dosya → Class Eşleşmesi

``` text
vectorch.py
└── VecTorch

manager.py
└── CollectionManager

collection.py
└── Collection

types.py
├── CollectionConfig
├── SearchResult
├── Record
├── Metric
└── IndexType

storage/engine.py
└── StorageEngine

storage/vectors.py
└── VectorStore

storage/ids.py
└── IDMap

storage/metadata.py
└── MetadataStore

storage/deleted.py
└── DeleteBitmap

index/base.py
└── Index (ABC)

index/flat.py
└── FlatIndex

index/hnsw.py
└── HNSWIndex

index/ivf.py
└── IVFIndex

kernels/base.py
└── DistanceKernel (ABC)

kernels/numpy.py
└── NumPyKernel

kernels/numba.py
└── NumbaKernel

kernels/native.py
└── NativeSIMDKernel

persistence/manager.py
└── PersistenceManager
```

------------------------------------------------------------------------

# 23. Önerilen Implementation Sırası

Mimarinin tamamını aynı anda kodlama.

``` text
1. types
       ↓
2. VectorStore
       ↓
3. IDMap
       ↓
4. MetadataStore + DeleteBitmap
       ↓
5. StorageEngine
       ↓
6. DistanceKernel ABC
       ↓
7. NumPyKernel
       ↓
8. Index ABC
       ↓
9. FlatIndex
       ↓
10. Collection
       ↓
11. CollectionManager
       ↓
12. VecTorch
       ↓
13. Persistence
       ↓
14. Benchmark baseline
       ↓
15. HNSW
       ↓
16. Numba kernel
       ↓
17. concurrency experiments
       ↓
18. native SIMD
       ↓
19. memory/cache optimization
       ↓
20. mmap / zero-copy
       ↓
21. IVF
       ↓
22. quantization / advanced HPC
```

------------------------------------------------------------------------

# 24. V1 Scope Sınırı

İlk çalışan V1 için minimum hedef:

``` text
VecTorch
CollectionManager
Collection
StorageEngine
VectorStore
IDMap
MetadataStore
DeleteBitmap
Index ABC
FlatIndex
DistanceKernel ABC
NumPyKernel
basic persistence
tests
benchmark baseline
```

**HNSW, IVF, Numba, SIMD, NUMA gibi özellikleri ilk çalışan sistem
ortaya çıkmadan ekleme.**

Önce doğru ve ölçülebilir baseline.

Sonra optimizasyon.

------------------------------------------------------------------------

# 25. Tasarımın Ana Prensipleri

Bu projede karar verirken şu kurallar referans alınmalı:

``` text
1. Correctness before optimization.

2. Hot-path'i küçük tut.

3. Vector data ile metadata'yı ayır.

4. Algoritmayı execution backend'den ayır.

5. Concrete implementation yerine interface'e bağımlı ol.

6. Gereksiz abstraction oluşturma.

7. Performans optimizasyonunu ölçmeden yapma.

8. Allocation ve data-copy maliyetlerini görünür hale getir.

9. Public API ile internal representation'ı birbirine bağlama.

10. Benchmark sonuçlarını reproducible yap.
```

------------------------------------------------------------------------

# 26. Tek Cümlelik Sistem Tanımı

**VecTorch, değiştirilebilir ANN index algoritmalarını değiştirilebilir
CPU execution kernel'leri üzerinde çalıştırabilen,
cache/memory/concurrency optimizasyonlarını deneysel olarak ölçmeye
uygun, embedded ve persistent bir vector search engine/database
olacaktır.**
