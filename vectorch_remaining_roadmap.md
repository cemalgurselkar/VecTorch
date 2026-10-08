# Vectorch --- Kalan Geliştirme Yol Haritası

Bu belge, mevcut Vectorch durumundan itibaren **Aşama 2'de kalan
işleri** ve sonraki geliştirme aşamalarını sıralar. Amaç coding agent'a
bütün projeyi yeniden tasarlatmak değil; mevcut mimari ve contract'ları
koruyarak sıradaki işleri uygulamasını sağlamaktır.

> Temel çalışma prensibi: Mimari ve public contract önce belirlenir.
> Coding agent bu tasarımı uygular. Bilerek geçici veya daha sonra
> tamamen yeniden yazılacak implementasyon yapılmaz. Her aşama test +
> mypy + ruff yeşil olmadan kapanmaz.

------------------------------------------------------------------------

# Mevcut durum

## Aşama 1 --- In-memory exact vector search

Tamamlandı.

Mevcut temel mimari:

``` text
Vectorch
  owns -> CollectionManager
             owns 0..N -> Collection
                           owns -> StorageEngine
                                    owns -> VectorStorage
                                    owns -> IDMap
                                    owns -> MetadataStorage
                                    owns -> DeleteBitmap
                           uses -> Index
                                   FlatIndex
                                   uses -> DistanceKernel
                                           NumpyKernel
                           uses -> PersistenceManager
```

Temel contract'lar:

-   Public facade: `Vectorch`
-   Vector internal dtype: `float32`
-   Vector layout: contiguous `(N, D)` NumPy array
-   External ID: `str | int`
-   Internal ID: stable integer ID
-   Metadata vector storage'dan ayrı
-   Delete: logical deletion / tombstone
-   `count()` yalnızca active kayıtlar
-   `total_count()` tombstone dahil tüm kayıtlar
-   Flat search exact baseline
-   L2 = squared Euclidean distance, düşük daha iyi
-   cosine/dot = yüksek daha iyi
-   public `get()` internal vector memory'sini expose etmez
-   metadata mutable leakage engellenir
-   lifecycle ve context manager uygulanmıştır

------------------------------------------------------------------------

# Aşama 2 --- Persistence

## Tamamlananlar

-   Persistence durability contract
-   Disk layout tasarımı
-   Binary serialization formatı
-   `vectors.bin`
-   `ids.bin`
-   `metadata.bin`
-   `deleted.bin`
-   `StorageSnapshot`
-   `StorageEngine.snapshot()`
-   `StorageEngine.from_snapshot()`
-   `PersistenceManager`
-   Database `manifest.json`
-   Generation-based snapshot yapısı
-   Atomic `CURRENT` publish
-   Basic corruption validation
-   Persistence format unit testleri
-   PersistenceManager unit testleri

Disk yapısı:

``` text
<db-root>/
├── manifest.json
└── collections/
    └── <collection-name>/
        ├── CURRENT
        └── snapshots/
            └── 0000000000000001/
                ├── config.json
                ├── vectors.bin
                ├── ids.bin
                ├── metadata.bin
                └── deleted.bin
```

Vector disk invariant:

``` text
dtype      = little-endian float32
shape      = (vector_count, dimension)
order      = C / row-major
layout     = contiguous
```

`CURRENT`, tamamen yazılmış snapshot generation'larından hangisinin
aktif olduğunu gösterir.

Durability contract:

``` text
mutation
   ↓
RAM state değişir
   ↓
collection dirty
   ↓
save() / close()
   ↓
atomic snapshot publish
   ↓
durable
```

Son başarılı `save()` öncesindeki crash durumunda yeni mutation'ların
kaybolması V1 contract'ında mümkündür. WAL/per-mutation durability henüz
yoktur.

------------------------------------------------------------------------

# Aşama 2 --- Tamamlanan entegrasyon işleri

Durum: **Tamamlandı.** Collection persistence entegrasyonu, startup
restore, dirty-state lifecycle, restart/drop davranışı, ortak filesystem
name contract'ı, JSON metadata ve signed-int64 external ID contract'ları
ile uçtan uca persistence testleri uygulanmıştır.

## 2.1 Collection persistence entegrasyonu

`Collection` persistence-aware hale getirilecek.

Gerekli davranışlar:

-   Collection bir `PersistenceManager` veya uygun persistence
    dependency almalı.
-   `_dirty: bool` state tutulmalı.
-   Yeni collection başlangıçta persistence açısından doğru state ile
    başlamalı.
-   Başarılı mutation sonrasında dirty olmalı:
    -   `add`
    -   `delete`
    -   gelecekte state değiştiren diğer operation'lar
-   `save()`:
    1.  collection açık mı kontrol et
    2.  storage snapshot al
    3.  `PersistenceManager.save_collection(...)`
    4.  yalnızca başarılı save sonrasında `_dirty = False`
-   Save başarısızsa dirty state korunmalı.
-   `close()` davranışı manager seviyesindeki lifecycle ile tutarlı
    olmalı.
-   Search/get/count gibi read-only operation'lar dirty state
    değiştirmemeli.

Public contract:

``` python
collection.save()
```

explicit durability boundary olarak kullanılabilmeli.

------------------------------------------------------------------------

## 2.2 Collection load

Diskten okunan:

``` text
CollectionConfig + StorageSnapshot
```

yeniden çalışan bir `Collection` oluşturabilmeli.

Restore sırasında:

-   internal ID sırası korunmalı
-   external ID mapping korunmalı
-   metadata korunmalı
-   deleted state korunmalı
-   metric/index config korunmalı
-   restore edilen collection dirty olmamalı
-   FlatIndex restore edilen StorageEngine üzerinde yeniden
    oluşturulmalı

FlatIndex'in kendisini ayrıca persist etmek gerekmez; Flat index vector
storage üzerinden yeniden çalışabilir.

------------------------------------------------------------------------

## 2.3 CollectionManager entegrasyonu

`CollectionManager`, `PersistenceManager` lifecycle'ının sahibi olacak.

Hedef:

``` text
Vectorch
   ↓
CollectionManager
   ├── RAM collection lifecycle
   └── PersistenceManager
          ↓
        disk lifecycle
```

Yapılacaklar:

-   `CollectionManager.__init__()` içinde `PersistenceManager(root)`
    oluştur.
-   Startup sırasında persisted collection'ları keşfet.
-   Her collection için config + snapshot load et.
-   Restore edilen collection'ları `_collections` içine ekle.
-   `create()` yeni collection yaratmalı fakat duplicate
    persisted/in-memory isimleri reddetmeli.
-   `drop()`:
    1.  collection'ı RAM lifecycle açısından kapat
    2.  disk persistence state'ini kaldır
    3.  manager registry'den kaldır
-   `list()` RAM/disk state ile tutarlı olmalı.
-   `close()` dirty collection'ları güvenli şekilde persist etmeli ve
    sonra kapatmalı.
-   Close idempotent kalmalı.

------------------------------------------------------------------------

## 2.4 Vectorch restart davranışı

Aşağıdaki kullanım çalışmalı:

``` python
with Vectorch(path) as db:
    docs = db.create_collection(
        "docs",
        dimension=3,
    )
    docs.add(
        "a",
        [1.0, 2.0, 3.0],
        {"source": "example"},
    )

with Vectorch(path) as db:
    docs = db.get_collection("docs")

    assert docs.count() == 1
```

`Vectorch(path)` aynı path açıldığında mevcut persisted database'i
keşfetmeli.

------------------------------------------------------------------------

## 2.5 Dirty-state ve lifecycle

Beklenen state machine kabaca:

``` text
loaded/created
    ↓
 clean
    ↓ mutation
 dirty
    ↓ save success
 clean
    ↓ mutation
 dirty
    ↓ close
 save
    ↓
 closed
```

Kurallar:

-   save success → clean
-   save failure → dirty
-   read-only operation → state değişmez
-   repeated close → güvenli/idempotent
-   stale Collection reference, DB close sonrası kullanılamaz
-   clean collection close edilirken gereksiz yeni snapshot yazılmamalı

Yeni collection'ın henüz hiç mutation almadan persistence davranışı açık
şekilde test edilmelidir. Collection creation'ın restart sonrası görünür
olması gerekiyorsa creation sırasında veya manager close sırasında ilk
empty snapshot persist edilmelidir. Contract test ile sabitlenmelidir.

------------------------------------------------------------------------

## 2.6 Drop persistence

Şu davranış garanti edilmeli:

``` python
db.drop_collection("docs")
db.close()

db = Vectorch(path)
assert "docs" not in db.list_collections()
```

Drop yalnızca RAM registry'den silmek değildir; persisted collection
state de kaldırılmalıdır.

Crash-safe drop için mevcut `PersistenceManager.drop_collection()`
atomic rename yaklaşımı korunmalıdır.

------------------------------------------------------------------------

## 2.7 Collection name filesystem contract

Collection name artık filesystem path'inin bir parçasıdır.

Path traversal engellenmelidir:

``` text
"../x"       reject
"."          reject
".."         reject
"a/b"        reject
"a\b"        reject
```

V1 naming contract tek bir yerde validate edilmelidir. Aynı validation
farklı katmanlarda farklı davranmamalıdır.

Gerekirse daha katı safe-name regex contract'ı mimari kararla
belirlenebilir; coding agent kendi başına public naming contract
genişletmemelidir.

------------------------------------------------------------------------

## 2.8 Metadata persistence contract

Metadata JSON tabanlı persist edildiği için metadata'nın persistent V1
contract'ı JSON-compatible olmalıdır.

Desteklenen temel değerler:

``` text
null
bool
number
string
list
object/dict
```

Arbitrary Python object veya pickle kullanılmamalıdır.

`NaN` / `Infinity` gibi standard JSON dışı numeric değerler persistence
boundary'de reddedilmelidir.

İdeal olarak hata `save()` sırasında çok geç çıkmak yerine metadata'nın
public mutation boundary'sinde validate edilmelidir. Bunun API etkisi
mimari olarak değerlendirilmelidir.

------------------------------------------------------------------------

## 2.9 External integer ID contract

`ids.bin` signed int64 encoding kullanır.

Dolayısıyla persistent V1 integer external ID contract:

``` text
-2^63 <= id <= 2^63 - 1
```

`bool`, Python'da `int` subclass olsa da external integer ID olarak
kabul edilmemelidir.

Bu constraint mümkünse `add()` boundary'sinde validate edilmelidir;
persistence sırasında sürpriz hata haline bırakılmamalıdır.

------------------------------------------------------------------------

## 2.10 End-to-end persistence testleri

En az şu davranışlar test edilmeli:

-   create → add → close → reopen → search
-   explicit `save()` → reopen
-   metadata restart sonrası aynı
-   external string ID restart sonrası aynı
-   external integer ID restart sonrası aynı
-   internal ID ordering restart sonrası korunuyor
-   deleted record restart sonrası deleted
-   deleted record search sonucunda yok
-   `count()` restart sonrası doğru
-   `total_count()` restart sonrası doğru
-   empty collection restart sonrası var
-   multiple collections restart sonrası var
-   collection config restart sonrası aynı
-   drop → restart sonrası collection yok
-   second save yeni generation publish ediyor
-   clean close gereksiz snapshot üretmiyor
-   corrupt `CURRENT` hata veriyor
-   eksik binary dosya hata veriyor
-   yanlış `vectors.bin` size hata veriyor
-   invalid manifest/version hata veriyor
-   save failure dirty state'i temizlemiyor

------------------------------------------------------------------------

## 2.11 Aşama 2 kalite kapısı

Aşama ancak şunların tamamı yeşil olduğunda kapanır:

``` bash
uv run pytest -v
uv run mypy src
uv run ruff check .
```

Ayrıca gerçek manual restart smoke test yapılmalıdır.

Aşama 2 sonunda beklenen özellik:

> Vectorch process kapatılıp yeniden açıldığında collection config,
> vectors, IDs, metadata ve tombstone state'i güvenli biçimde geri
> yüklenir.

------------------------------------------------------------------------

# Aşama 3 --- Benchmark Baseline ve Profiling

Durum: **Araçlar hazır, ölçüm çalıştırması bekleniyor.** Tekrarlanabilir
quick/standard workload'ları, JSON baseline raporu, exact NumPy oracle,
latency/QPS/ingestion/persistence/RSS ölçümleri, cProfile hot-path aracı
ve baseline karşılaştırıcı eklenmiştir. Kalite kapısı ve gerçek baseline
çalıştırması kullanıcı tarafından yapılacaktır.

Standard baseline hedefi 1.000.000 vector, 128 dimension ve 1.000 ölçülen
query olarak sabitlenmiştir. Full-sort doğruluk oracle'ı bu ölçekte sabit
seed ile seçilen 25 query üzerinde çalışır.

Scaling baseline tek bir 1M noktası yerine 10K, 25K, 50K, 100K, 250K,
500K ve 1M aşamalarını ayrı process'lerde ölçer. Her aşama kendi JSON
raporunu üretir ve toplu suite manifest'ine eklenir.

Persistence tamamlandıktan sonra hemen optimizasyon yapılmamalı. Önce
ölçüm altyapısı kurulmalı.

Amaç:

``` text
Measure → Profile → Hypothesis → Optimize → Measure again
```

Benchmark kapsamı:

-   dataset size
-   dimension
-   metric
-   k
-   query count
-   warmup
-   thread count
-   machine/CPU bilgisi
-   reproducible random seed

Ölçümler:

-   p50 latency
-   p95 latency
-   p99 latency
-   p99.9 latency
-   QPS
-   memory/RSS
-   index/build time
-   Recall@K (ANN aşamasında)
-   CPU utilization

FlatIndex exact ground truth olarak kullanılmalı.

Benchmark kodu production library kodundan ayrı tutulmalı:

``` text
benchmarks/
experiments/
```

Bu aşamada NumPy baseline ölçülür. Ölçüm olmadan Numba/SIMD
optimizasyonuna geçilmez.

------------------------------------------------------------------------

# Aşama 4 --- DistanceKernel Performance Layer

Mevcut abstraction:

``` text
DistanceKernel
├── NumpyKernel
├── NumbaKernel
└── NativeSIMDKernel
```

Önce NumPy baseline profile edilir.

Ardından ölçüm gerekçesi varsa:

## NumbaKernel

-   cosine
-   dot
-   squared L2
-   allocation azaltma
-   parallel execution yalnızca benchmark ile gerekçelendirilirse

Correctness bütün kernel'lerde aynı olmalı.

Aynı query/vector input için ranking ve score semantics korunmalı.

## Native SIMD

Numba sonrası hâlâ anlamlı bottleneck varsa native C/C++ extension
düşünülebilir:

-   AVX2
-   AVX-512 uygun hardware'de
-   FMA
-   aligned/contiguous access
-   GIL release

Native katman ilk çözüm değildir; benchmark sonucu
gerekçelendirilmelidir.

------------------------------------------------------------------------

# Aşama 5 --- Search Hot-Path ve Memory Optimization

Profiling sonucuna göre:

-   cosine vector norm caching/precomputation
-   L2 temporary allocation azaltma
-   reusable search buffers
-   `SearchContext`
-   top-k allocation azaltma
-   cache locality
-   unnecessary copies
-   vector alignment
-   batch operations

Önemli prensip:

> Python object ve metadata hot path'e sokulmamalıdır.

Search akışı:

``` text
query
  ↓
numeric vector/index layer
  ↓
top-k internal IDs
  ↓
ID + metadata resolution
  ↓
SearchResult
```

korunmalıdır.

------------------------------------------------------------------------

# Aşama 6 --- Batch Search ve Concurrency

Önce API contract tasarlanmalı.

Muhtemel batch shape:

``` text
queries: (B, D)
```

Flat exact batch için matrix-matrix yaklaşımı değerlendirilebilir:

``` text
Q @ V.T
```

Concurrency hedefi başlangıçta:

``` text
single writer
multiple readers
```

Değerlendirilecek modeller:

-   query-level parallelism
-   intra-query parallelism
-   thread pool/native threads
-   GIL etkisi
-   read/write locking
-   immutable snapshots / copy-on-write gerekirse
-   false sharing
-   thread-local scratch buffers

Thread count arttırmanın otomatik olarak performansı artırdığı
varsayılmamalıdır. Memory bandwidth saturation benchmark edilmelidir.

------------------------------------------------------------------------

# Aşama 7 --- mmap / Persistence I/O Optimization

Mevcut `vectors.bin` layout mmap'i engellememektedir.

Bu aşamada:

``` text
read entire vectors.bin → RAM
```

ile:

``` text
memory map vectors.bin
```

karşılaştırılır.

Ölçülecekler:

-   startup latency
-   RSS
-   page faults
-   cold search latency
-   warm search latency
-   sequential scan throughput

Page cache davranışı incelenmelidir.

`mmap` yalnızca "daha hızlıdır" varsayımıyla kullanılmamalıdır.

------------------------------------------------------------------------

# Aşama 8 --- HNSW Index

Flat exact baseline stabil olduktan sonra ilk ANN index.

`Index` interface gerçek HNSW ihtiyaçlarına göre genişletilir; önceden
tahmin ederek gereksiz method eklenmez.

HNSW için:

-   graph representation
-   insertion
-   search
-   `M`
-   `ef_construction`
-   `ef_search`
-   deletion interaction
-   persistence
-   memory usage
-   build time
-   Recall@K / latency trade-off

FlatIndex ground truth sağlar.

Karşılaştırma:

``` text
same dataset
same dimensions
same metric
same k
same hardware
same thread count
same recall target
```

------------------------------------------------------------------------

# Aşama 9 --- IVF Index

HNSW sonrasında ikinci ANN strategy.

Konular:

-   coarse clustering
-   centroid training
-   inverted lists
-   `nlist`
-   `nprobe`
-   build/train lifecycle
-   persistence
-   Recall@K / latency
-   memory footprint

Index selection V1'de explicit kalır:

``` python
index="flat"
index="hnsw"
index="ivf"
```

`index="auto"` ancak ileride gerçek benchmark verisine dayanarak
düşünülebilir.

------------------------------------------------------------------------

# Aşama 10 --- Compaction

Logical deletion nedeniyle tombstone'lar RAM ve disk alanı kullanmaya
devam eder.

Compaction:

``` text
vectors + IDs + metadata + deleted
              ↓
       active records
              ↓
      new compact state
```

Ancak internal ID remapping index'leri etkiler.

Bu nedenle compaction tasarımı:

-   atomic
-   index-aware
-   crash-safe
-   measurable

olmalıdır.

Compaction threshold tahminle değil ölçümle seçilmelidir.

------------------------------------------------------------------------

# Aşama 11 --- Stronger Durability / WAL

Snapshot durability yeterli değilse WAL eklenebilir.

Hedef:

``` text
mutation
  ↓
WAL durable
  ↓
memory state
  ↓
background/checkpoint snapshot
```

Konular:

-   log record format
-   sequence numbers
-   replay
-   torn/corrupt records
-   checksum
-   checkpoint
-   WAL truncation
-   fsync policy
-   group commit

Bu aşama snapshot persistence'ın yerine geçmek zorunda değildir;
snapshot + WAL birlikte kullanılabilir.

------------------------------------------------------------------------

# Aşama 12 --- Quantization / Memory Efficiency

Profiling ve ANN baseline sonrasında:

-   FP16
-   INT8
-   scalar quantization
-   product quantization (PQ)

değerlendirilebilir.

Her optimizasyon:

``` text
memory saving
vs
latency
vs
Recall@K / accuracy
```

üçgeninde ölçülmelidir.

`float32` canonical baseline olarak korunmalıdır.

------------------------------------------------------------------------

# Aşama 13 --- Advanced HPC Engineering

Yalnızca ölçüm gerekçesi varsa:

-   CPU affinity
-   NUMA awareness
-   cache miss profiling
-   hardware performance counters
-   prefetching
-   false-sharing analysis
-   huge pages değerlendirmesi
-   native thread pools
-   vectorized top-k
-   memory alignment
-   page-fault analysis

Araçlar/platform uygunluğuna göre:

``` text
perf
py-spy
Linux hardware counters
benchmark reports
```

kullanılabilir.

------------------------------------------------------------------------

# Aşama 14 --- External Engine Benchmarking

Vectorch ancak kendi benchmark sistemi stabil olduktan sonra dış
motorlarla karşılaştırılmalıdır.

Adaylar:

-   FAISS
-   hnswlib
-   Qdrant

Adil karşılaştırma şartları:

``` text
same dataset
same metric
same dimensions
same k
same hardware
same threads
same warm/cold condition
same Recall@K target
```

Amaç "Vectorch daha hızlı" sonucu üretmek değil; hangi workload'da neden
hızlı/yavaş olduğunu gösterebilmektir.

------------------------------------------------------------------------

# Aşama 15 --- Documentation ve Portfolio Packaging

README yalnızca özellik listesi olmamalıdır.

Dokümantasyon:

-   problem definition
-   architecture diagram
-   storage layout
-   persistence/durability semantics
-   index architecture
-   benchmark methodology
-   benchmark results
-   profiling findings
-   optimizations ve nedenleri
-   trade-offs
-   limitations
-   reproducibility instructions

Özellikle portfolio açısından şu hikâye görünür olmalıdır:

``` text
correct baseline
    ↓
measurement
    ↓
identified bottleneck
    ↓
engineering hypothesis
    ↓
optimization
    ↓
measured result
```

------------------------------------------------------------------------

# Coding Agent İçin Kurallar

Coding agent mevcut mimariyi kendi başına yeniden tasarlamamalıdır.

Her görevde:

1.  Önce ilgili mevcut kodu ve testleri oku.
2.  Mevcut public contract'ı koru.
3.  Mimari karar belirsizse implement etmeden önce sor.
4.  Bilerek temporary implementation yazma.
5.  Gereksiz abstraction ekleme.
6.  Private implementation detaylarını katmanlar arasında sızdırma.
7.  Correctness'i performanstan önce doğrula.
8.  Her değişiklik için davranış odaklı test ekle.
9.  Değişiklik sonunda çalıştır:

``` bash
uv run pytest -v
uv run mypy src
uv run ruff check .
```

10. Testler yeşil olmadan görevi tamamlanmış sayma.
11. Benchmark sonucu olmadan "optimization" yapma.
12. Persistence/index/storage gibi format veya public contract
    değişikliklerini sessizce yapma; önce gerekçeyi açıkla.

------------------------------------------------------------------------

# Geliştirme metodolojisi

Her yeni sistem bileşeninde:

``` text
Software design
    ↓
Dependency graph
    ↓
Least-dependent component
    ↓
Responsibility
    ↓
What it must NOT do
    ↓
Public interface
    ↓
Internal representation
    ↓
Invariants / edge cases
    ↓
Tests
    ↓
Correct implementation
    ↓
Tests
    ↓
Next layer
    ↓
Benchmark / profile
    ↓
Optimization
```

Ana performans metodolojisi:

``` text
Measure
  ↓
Profile
  ↓
Hypothesis
  ↓
Optimize
  ↓
Measure again
```

Bu iki döngü Vectorch geliştirme sürecinin temelidir.
