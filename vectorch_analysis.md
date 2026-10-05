# Vectorch — Vector Search Engine Durum Analizi

## Yönetici özeti

Projede çalışan bir dikey arama akışı vardır; ancak henüz uçtan uca, kalıcı bir vector search engine tamamlanmış değildir.

Mevcut durum en doğru ifadeyle **in-memory exact/flat vector search prototipi** olarak tanımlanabilir.

Sistem şu anda bir vector ekleyip en yakın kayıtları bulabilmektedir. Buna karşılık engine kapatılıp yeniden açıldığında verinin korunması, approximate nearest neighbor (ANN) araması, metadata filtering, batch ingestion, update/upsert ve güvenilir veri yaşam döngüsü henüz bulunmamaktadır.

Bu nedenle Numba, SIMD veya ileri ANN optimizasyonlarından önce şu dört alanın tamamlanması önerilir:

1. API ve doğruluk sözleşmesi
2. Basic persistence
3. Doğruluk testleri ve brute-force oracle
4. Tekrarlanabilir benchmark altyapısı

---

## Mevcut mimari ve çalışan akış

```text
Vectorch
  → CollectionManager
  → Collection
  → StorageEngine
      → VectorStorage
      → IDMap
      → MetadataStorage
      → DeleteBitmap
  → FlatIndex
  → NumpyKernel
  → SearchResult
```

Şu özellikler gerçekten çalışmaktadır:

- Collection oluşturma
- Tek tek vector ekleme
- External ID ile internal ID arasında eşleme
- Metadata saklama
- Cosine, dot product ve squared-L2 hesaplama
- Exact top-k arama
- Logical deletion
- Silinen vectorleri arama sonuçlarından çıkarma
- External ID ve metadata ile sonuç döndürme
- Boş collection üzerinde arama
- `k` değerinin collection boyutundan büyük olması

İnceleme sırasında mevcut 17 testin tamamı geçmiştir.

Bu nedenle “bir vector verip en yakın kayıtları alabiliyor muyum?” sorusunun cevabı **evet**tir. Ancak persistence, ANN, filtreleme ve güvenilir veri yönetimi açısından sistem henüz tamamlanmış değildir.

---

## Kritik eksikler

### 1. Persistence bulunmuyor

En kritik eksik persistence katmanıdır.

`Vectorch` bir `path` almakta ve `CollectionManager` bu path'i saklamaktadır; fakat bu konumdan herhangi bir veri okunmamakta veya buraya veri yazılmamaktadır. `Collection.close()` metodu da boş durumdadır.

Doğrulanan davranış:

```text
collection oluştur
vector ekle
db.close()
aynı path ile tekrar aç
→ collection listesi boş
```

Tasarım belgesinde V1'in embedded ve persistent olması, ayrıca basic persistence içermesi hedeflenmiştir. Dolayısıyla mevcut implementasyon kendi V1 kapsamını henüz tamamlamamaktadır.

Gerekli temel parçalar:

- Collection manifest ve config dosyaları
- Vector buffer serialization
- ID map persistence
- Metadata persistence
- Delete bitmap persistence
- Index state persistence
- `CollectionManager.load_all()`
- Atomik save/flush
- Yarım veya bozuk yazılmış snapshot yönetimi
- Disk formatı için version bilgisi
- Restart ve recovery testleri

### 2. Yalnızca exact FlatIndex var

`Collection` yalnızca `FlatIndex` oluşturabilmektedir. HNSW ve IVF enum değerleri tanımlanmış olsa da implementasyonları bulunmamaktadır.

Mevcut sistem:

- Her sorguda bütün vectorleri tarar.
- Yaklaşık `O(N × dimension)` search maliyetine sahiptir.
- Büyük RAG corpuslarında doğrusal olarak pahalılaşır.
- Henüz gerçek bir ANN engine değildir.

Bu durum başlangıç aşaması için yanlış değildir ve tasarım belgesindeki implementation sırasıyla uyumludur. Yine de HNSW veya IVF implementasyonundan önce persistence ve benchmark baseline tamamlanmalıdır.

### 3. Metadata filtering bulunmuyor

Metadata saklanmakta ve arama sonucunda döndürülmektedir; fakat sorgu sırasında filtre olarak kullanılamamaktadır.

Tasarım belgesinde aşağıdaki API öngörülmektedir:

```python
search(query, k, filter=None)
```

Mevcut `Collection.search()` ise filtre kabul etmemektedir.

RAG sistemlerinde ihtiyaç duyulabilecek tipik filtreler:

- Tenant
- Document veya source
- Tarih aralığı
- Access control
- Dil
- Kategori
- Chunk/document ilişkisi

ANN eklendiğinde pre-filter ve post-filter stratejileri de ayrıca tasarlanmalıdır.

### 4. Query ve configuration doğrulaması eksik

Vector ekleme tarafında boyut ve `float32` kontrolü vardır. Search query için eşdeğer bir doğrulama yoktur. Query doğrudan distance kernel'e iletilmektedir.

Mevcut davranışlar:

- `float64` query sessizce kabul edilir.
- 2D query, public API seviyesinde anlamlı bir hata yerine NumPy `matmul` hatası üretir.
- `NaN` içeren query, `NaN` skorlar üretir.
- Geçersiz metric collection oluşturulurken değil, ilk search sırasında hata verir.
- String metric/index değerleri runtime'da enum'a dönüştürülmez.

Public API sınırında aşağıdaki kontroller yapılmalıdır:

- Query shape tam olarak `(dimension,)` olmalı
- Dtype politikası açık olmalı: strict `float32` veya kontrollü dönüşüm
- `NaN` ve `inf` politikası belirlenmeli
- Metric/index string değerleri enum'a dönüştürülmeli
- `k` için integer ve bool ayrımı kontrol edilmeli
- Contiguous memory gereksinimi açıkça belirlenmeli

### 5. Dahili mutable veri dışarı sızıyor

`VectorStorage.get()` doğrudan dahili NumPy satırını döndürmektedir. Bu nedenle aşağıdaki işlem database içindeki vectorü de değiştirmektedir:

```python
vector = collection.get("a")
vector[:] = 0
```

Metadata tarafında da benzer bir durum vardır. Metadata dict'i kopyalanmadan saklanmakta ve sonuçlarda aynı nesne döndürülmektedir. Search sonucundaki metadata değiştirildiğinde storage içindeki metadata da değişmektedir.

Bu davranış bilinçli bir zero-copy API değilse doğruluk problemidir.

Önerilen yaklaşım:

- Güvenli public API kopya veya read-only değer döndürmeli.
- Zero-copy erişim gerekiyorsa ayrı ve açıkça `unsafe` veya `view` olarak adlandırılan bir API sunulmalı.

### 6. Delete ve count semantiği net değil

Delete işlemi fiziksel silme yerine tombstone/logical deletion kullanmaktadır. Bu başlangıç için doğru bir tercihtir. Fakat mevcut `count()` silinmiş kayıtları da saymaktadır.

Örnek:

```text
1 kayıt ekle
kaydı sil
count() → 1
```

Şu ayrımın yapılması önerilir:

- `active_count`
- `total_count`
- `deleted_count`

Ayrıca silinen bir external ID tekrar eklenememektedir. Bu davranış update/upsert ve tombstone politikasıyla birlikte tanımlanmalıdır.

`Collection.search()` içindeki, silinen kayıtların top-k öncesinde filtrelenmesi gerektiğini belirten TODO artık eskidir. `FlatIndex` silinen kayıtları top-k seçiminden önce zaten filtrelemektedir.

### 7. Index interface gelecekteki indexler için yetersiz

Mevcut `Index` abstract interface yalnızca `search()` metodunu içermektedir.

Flat index var olan vector bufferını doğrudan taradığı için `add()` metoduna ihtiyaç duymamaktadır. HNSW ve IVF gibi stateful indexler için ise muhtemelen şu sözleşmeler gerekecektir:

- `add(internal_ids)`
- `remove(internal_ids)`
- `build()`
- `search()`
- `save()` / `load()`
- `close()`
- Index-specific stats

Collection şu anda `StorageEngine.add()` çağırmakta ancak index'e ekleme bildirimi göndermemektedir. HNSW eklendiğinde bu orchestration akışının genişletilmesi gerekecektir.

### 8. Batch API bulunmuyor

RAG ingestion sırasında vectorlerin tek tek Python çağrısıyla eklenmesi önemli bir darboğaz oluşturacaktır.

Mevcut API:

```python
collection.add(id, vector, metadata)
```

Eksik olan toplu operasyonlar:

- `add_many`
- `upsert_many`
- `delete_many`
- `get_many`
- `search_batch`

Mevcut 10.000 vector testi bile Python döngüsüyle insert yapmaktadır. Batch ingestion, ilk anlamlı performans çalışmalarından biri olmalıdır.

### 9. Atomiklik ve concurrency bulunmuyor

`StorageEngine.add()` vector, ID ve metadata yapılarını ayrı ayrı güncellemektedir. Adımlardan biri hata verirse rollback mekanizması yoktur.

Henüz aşağıdaki yapılar bulunmamaktadır:

- Lock veya reader/writer politikası
- Snapshot isolation
- Transaction/rollback
- Concurrent add/search
- Crash recovery

Concurrency ilk aşamada uygulanmak zorunda değildir. Ancak persistence formatı ve mutation modeli ileride concurrency eklenebileceği düşünülerek tasarlanmalıdır.

---

## Mevcut performans darboğazları

### Cosine her sorguda bütün normları yeniden hesaplıyor

Cosine implementasyonunda her sorguda tüm corpus için aşağıdaki işlem yapılmaktadır:

```python
vector_norms = np.linalg.norm(vectors, axis=1)
```

Corpus vector normları vector eklenirken bir kez hesaplanıp saklanabilir. Alternatif olarak cosine collection içindeki vectorler normalize edilmiş biçimde tutulabilir.

Bu, mevcut FlatIndex'teki en kolay ve anlamlı performans kazanımlarından biri olabilir.

### L2 tam `N × D` temporary allocation oluşturuyor

Mevcut L2 hesabı:

```python
diff = vectors - query
np.sum(diff * diff, axis=1)
```

Büyük corpuslarda tam boyutlu temporary matrix oluşturur; `diff * diff` de ikinci bir temporary oluşturabilir.

Olası alternatifler:

- Chunked calculation
- `||x||² + ||q||² - 2x·q`
- Buffer reuse
- Optimize edilmiş Numba/native kernel

Cebirsel form kullanılırsa floating-point doğruluğu ve cancellation etkisi ayrıca ölçülmelidir.

### Flat search memory bandwidth-bound olacaktır

Exact search büyüdükçe şu faktörler belirleyici olacaktır:

- Memory bandwidth
- Cache locality
- Vector normalization ve memory layout
- Batch query ile GEMM kullanımı
- Top-k selection
- Chunk size

Doğrudan SIMD veya Numba implementasyonuna geçmeden önce bu noktalar ayrı benchmarklarla ölçülmelidir.

---

## API ve proje düzeni uyuşmazlıkları

Tasarım dokümanı ile kod arasında şu uyuşmazlıklar bulunmaktadır:

- Dokümanda `VecTorch`, kodda `Vectorch`
- Dokümanda `list_collections`, kodda `list_collection`
- Dokümanda `get()` bir `Record` döndürüyor, kodda yalnızca vector dönüyor
- Dokümanda batch `add`, kodda tekil add bulunuyor
- `Record` tipi tanımlı fakat kullanılmıyor
- README boş
- Package description hâlâ `"Add your description here"`
- Numba dependency olarak ekli fakat kullanılmıyor
- `benchmarks/`, `experiments/` ve `native/` dizinleri boş

Bu uyuşmazlıklar optimizasyondan önce çözülmelidir; aksi takdirde performans çalışmaları değişken bir API ve veri modeli üzerinde yapılmış olur.

---

## Statik analiz ve test sonuçları

İnceleme sırasında elde edilen sonuçlar:

| Kontrol | Sonuç |
|---|---:|
| Pytest | 17 test geçti |
| Mypy | 4 hata |
| Ruff | 11 hata |

Mypy hataları arasında gerçek API tipi problemleri bulunmaktadır:

- Public API string metric kabul ediyor, `CollectionConfig` ise `Metric` bekliyor.
- Public API string index kabul ediyor, `CollectionConfig` ise `IndexType` bekliyor.
- `StorageEngine.get_vector()` return annotation'ı yanlışlıkla dict olarak belirtilmiş.
- Bu yanlış annotation, `Collection.get()` üzerinde ikinci bir tip hatasına neden oluyor.

Ruff bulguları import sıralaması ve format düzeniyle ilgilidir.

Mevcut testler temel happy path'i doğrulasa da aşağıdaki konuları kapsamamaktadır:

- Persistence/restart
- Randomized brute-force karşılaştırması
- Invalid metric/index
- Query dtype ve shape doğrulaması
- `NaN`/`inf`
- Zero-vector cosine davranışı
- Tie ordering
- Delete sonrası count
- Silinen ID'nin tekrar eklenmesi
- Mutable vector/metadata erişimi
- Partial write/rollback
- Corrupt veya truncated persistence dosyaları
- Concurrent erişim

---

## Optimizasyondan önce önerilen çalışma sırası

### Aşama 1 — Public contract ve doğruluk

- API isimlerini sabitle
- Metric ve index değerlerini normalize et
- Query validation ekle
- Score/distance semantiğini belgele
- Delete/count/upsert davranışını belirle
- Mutable view politikasını belirle
- Mypy ve Ruff kontrollerini temizle

### Aşama 2 — Basic persistence

- Versioned manifest
- Collection config save/load
- Vector, ID, metadata ve tombstone serialization
- Atomik snapshot
- Reopen/load flow
- Restart ve recovery testleri

### Aşama 3 — Doğruluk testleri

- Randomized brute-force oracle
- Restart testleri
- Invalid input testleri
- Delete + restart
- Duplicate/upsert
- Zero-vector cosine
- Tie davranışı
- Capacity growth
- Corrupt file senaryoları

FlatIndex, sonraki ANN indexlerinin doğruluk referansı olarak tutulmalıdır.

### Aşama 4 — Benchmark baseline

En az şu metrikler ölçülmelidir:

- Insert throughput
- p50, p95 ve p99 query latency
- QPS
- Peak memory
- Build time
- Farklı corpus boyutları
- Farklı dimension değerleri
- Farklı `k` değerleri
- Recall@k
- Cold ve warm çalışma farkı

Benchmark veri setleri ve random seed sabitlenmeli, sonuçlar makine bilgisiyle birlikte kaydedilmelidir.

### Aşama 5 — FlatIndex optimizasyonu

Ölçüm sonuçlarına göre:

- Cosine pre-normalization veya stored norms
- Batch add/search
- L2 allocation azaltma
- Chunked search
- Top-k algoritması benchmarkları
- Memory layout deneyleri

### Aşama 6 — HNSW

İlk custom ANN index olarak HNSW mantıklı bir seçimdir.

HNSW değerlendirmesinde yalnızca latency değil, şu dört eksen birlikte ölçülmelidir:

- Recall@k
- Query latency/QPS
- Build time
- Memory consumption

FlatIndex aynı veri setinde ground-truth üretmeye devam etmelidir.

### Aşama 7 — RAG özellikleri ve sistem davranışı

- Metadata filtering
- Update/upsert
- Tombstone compaction
- Concurrent readers/writers
- Mmap/zero-copy
- Crash recovery
- IVF/quantization gibi alternatif indexler

---

## Sonuç

Projede storage'dan exact search ve top-k sonucuna kadar uzanan, mimariyi doğrulayan iyi bir başlangıç baseline'ı vardır.

Ancak mevcut haliyle sistem:

- In-memory çalışır.
- Yalnızca exact flat search yapar.
- Restart sonrasında veriyi geri yüklemez.
- Metadata filtering sunmaz.
- Batch ingestion/query sunmaz.
- ANN index içermez.
- Henüz kapsamlı doğruluk ve performans ölçüm altyapısına sahip değildir.

Bu nedenle sistem henüz genel amaçlı, uçtan uca bir RAG vector search engine değildir. Optimizasyondan önce **API doğruluğu, persistence, test oracle ve benchmark altyapısı** tamamlanmalıdır. Bu dört temel tamamlandıktan sonra yapılacak FlatIndex, HNSW, Numba veya SIMD çalışmaları ölçülebilir ve güvenilir hale gelecektir.
