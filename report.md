# Vectorch Teknik Teşhis Raporu

Bu belge, optimizasyon öncesi mevcut Vectorch uygulamasının mimarisini,
doğruluk durumunu, 10 bin ile 1 milyon vektör arasında alınan benchmark
sonuçlarını ve profilleme bulgularını kayda geçirir. Amaç tek bir sayıyı
tanıtım verisine dönüştürmek değil, sonraki optimizasyonlar için tekrar
üretilebilir bir başlangıç noktası oluşturmaktır.

## 1. Mevcut sistem

Vectorch, Python uygulamasının içinde çalışan kalıcı bir vektör
veritabanıdır. Ayrı bir sunucu veya ağ servisi gerektirmez. Mevcut `0.1.0`
sürümü şu özellikleri sağlar:

- Bir veritabanı yolu altında birden fazla collection.
- `str` veya signed 64-bit `int` harici kimlikler.
- JSON uyumlu metadata.
- `float32`, C-contiguous dahili vektör saklama.
- Cosine similarity, dot product ve squared L2 distance.
- NumPy tabanlı exact `FlatIndex` araması.
- Mantıksal silme (tombstone), `count()` ve `total_count()` ayrımı.
- Generation tabanlı snapshot persistence ve atomik `CURRENT` yayını.
- Yeniden başlatıldığında collection, ID, metadata ve silinme durumunu
  geri yükleme.

Yüksek seviyeli sahiplik zinciri şöyledir:

```text
Vectorch
  └─ CollectionManager
       ├─ PersistenceManager
       └─ Collection
            ├─ StorageEngine
            │    ├─ VectorStorage
            │    ├─ IDMap
            │    ├─ MetadataStorage
            │    └─ DeleteBitmap
            └─ FlatIndex
                 └─ NumpyKernel
```

Bu sürüm bir ANN motoru değildir. Her sorgu bütün aktif veri kümesini
tarar; dolayısıyla arama karmaşıklığı yaklaşık `O(N × D)`'dir. Bunun
avantajı exact sonuç ve optimizasyonları ölçmek için güvenilir bir baseline,
dezavantajı ise veri sayısıyla yaklaşık doğrusal artan sorgu maliyetidir.

## 2. Persistence ve dayanıklılık sözleşmesi

Her collection diskte generation numaralı snapshot'larla saklanır. Yeni
snapshot önce geçici bir dizine tamamen yazılır, dosyalar ve dizinler
`fsync` edilir, sonra snapshot atomik olarak yerine alınır ve en son `CURRENT`
göstergesi yayınlanır. Bu tasarım, yarım yazılmış yeni bir snapshot'ın aktif
snapshot olarak görülmesini engeller.

Dayanıklılık sınırı şudur:

```text
add/delete -> RAM değişir -> collection dirty olur
           -> save() veya close() -> snapshot kalıcı olur
```

Henüz WAL ve her mutation sonrası otomatik disk garantisi yoktur. Son başarılı
`save()` veya `close()` sonrasında yapılıp diske yazılmayan değişiklikler süreç
çökerse kaybolabilir. Bu, mevcut sürümün bilinçli sözleşmesidir.

## 3. Benchmark yöntemi

Kademeli suite şu boyutları ayrı Python subprocess'lerinde çalıştırdı:

```text
10K -> 25K -> 50K -> 100K -> 250K -> 500K -> 1M
```

Her aşama bittiğinde süreç ve geçici veritabanı kapatıldı; böylece Python
nesneleri, NumPy buffer'ları, allocator state'i ve process RSS'i bir sonraki
aşamaya taşınmadı. Ayrıcalıklı ve makine genelini etkileyen bir işlem olduğu
için işletim sistemi page cache'i zorla temizlenmedi.

Ortak workload:

| Parametre | Değer |
|---|---:|
| Dimension | 128 |
| Metric | cosine |
| K | 10 |
| Ölçülen sorgu | 1.000 |
| Warmup sorgusu | 50 |
| Oracle sorgusu | 25 |
| Metadata | kapalı |
| Seed | 42 |

Ortam:

| Alan | Değer |
|---|---|
| OS | Linux 6.8, x86_64, glibc 2.35 |
| Mantıksal CPU | 8 |
| Python | CPython 3.11.16 |
| NumPy | 2.4.6 |
| Git commit | `7545752f106a6151c7efe751dffe071eabce2ae3` |
| Worktree | dirty |
| BLAS/OpenMP thread env | Hepsi ayarlanmamış |

Dirty worktree notu önemlidir: Bu sonuçlar commit'in tek başına yeniden
oluşturduğu bir release sonucu değil, geliştirme ağacının baseline'ıdır.

## 4. Kademeli arama sonuçları

| Vektör | p50 (ms) | p99 (ms) | QPS |
|---:|---:|---:|---:|
| 10.000 | 1,406 | 5,269 | 519,97 |
| 25.000 | 4,097 | 10,082 | 202,47 |
| 50.000 | 8,956 | 15,816 | 103,51 |
| 100.000 | 19,969 | 27,615 | 48,59 |
| 250.000 | 59,913 | 117,062 | 15,59 |
| 500.000 | 111,326 | 150,390 | 8,75 |
| 1.000.000 | 199,701 | 285,300 | 4,88 |

Bir milyon vektör aşamasında ek değerler:

- Ortalama: 204,790 ms.
- p95: 245,042 ms; p99.9: 309,546 ms; maksimum: 335,983 ms.
- Warmup öncesi ilk sorgu: 261,521 ms.
- 1.000 sorgunun toplam ölçüm süresi: 204,795 saniye.
- Arama CPU süresi: 957,423 saniye; ortalama CPU kullanımı: `%467,5`.

Eğri, exact flat taramanın beklendiği gibi veri sayısıyla kuvvetli biçimde
ölçeklendiğini gösteriyor. 10K'dan 1M'e veri 100 kat artarken p50 yaklaşık
142 kat artıyor ve QPS yaklaşık 106 kat azalıyor. Bu tek başına bir hata
değil; mevcut algoritmanın maliyet modelinin sonucu. Doğrusal çizgiden ek
sapma ise cache davranışı, bellek bant genişliği, allocation ve thread
verimliliğiyle uyumludur.

## 5. Doğruluk

Bağımsız NumPy oracle ile kontrol edilen 25 sorgu ve 250 skor için:

- Mean Recall@10: `1.0`.
- Minimum Recall@10: `1.0`.
- Exact top-k set match: `1.0`.
- Ortalama ve maksimum mutlak skor hatası: `0.0`.

Bu sonuç mevcut exact baseline'ın test edilen cosine workload'unda doğru
olduğunu gösterir. Tüm olası veri dağılımları için matematiksel kanıt veya ANN
kalite iddiası değildir.

## 6. Ingestion, disk ve bellek

Bir milyon vektör aşamasında:

- Ingestion: 6,339 saniye, yaklaşık 157.746 vektör/saniye.
- Snapshot save: 3,502 saniye.
- Yeniden açma: 4,869 saniye.
- Veritabanı boyutu: 530.000.286 byte, yaklaşık 530 byte/vektör.
- Dataset oluşturulduktan sonra RSS: yaklaşık 533 MiB.
- Ingestion sonrası RSS: yaklaşık 1.143 MiB.
- Yeniden açma sonrası RSS: yaklaşık 1.193 MiB.
- Arama/oracle sonrası process peak RSS: yaklaşık 2.268 MiB.

128 boyutlu bir milyon ham `float32` vektör yaklaşık 488 MiB'dir. Dataset,
storage, snapshot/reopen ve arama sırasındaki geçici dizilerin aynı anda
yaşaması peak RSS'in bunun birkaç katına çıkmasını açıklar. Benchmark
metadata'yı bilerek kapalı tuttuğu için bu sayılar metadata ağır bir RAG
workload'unun bellek maliyetini temsil etmez.

## 7. Profilleme teşhisi

Bir milyon vektörde yalnızca 1.000 measured search'ü kapsayan `cProfile`
özetinde toplam süre 168,288 saniyedir:

| Hot path | Kümülatif süre | Profil payı |
|---|---:|---:|
| `Collection.search` | 168,288 s | ~%100 |
| `FlatIndex.search` | 168,173 s | ~%99,9 |
| `NumpyKernel.compute` | 161,154 s | ~%95,8 |
| `np.linalg.norm` | 141,176 s | ~%83,9 |
| `argpartition` | 4,252 s | ~%2,5 |

Ana darboğaz top-k seçimi veya Python'da `SearchResult` oluşturmak değil,
cosine kernel'idir. Mevcut kernel her sorguda:

1. `vectors @ query` hesaplar.
2. Bütün corpus vektörlerinin normunu baştan hesaplar.
3. Query normunu hesaplar.
4. Yeni denominator ve skor dizileri oluşturur.

Corpus vektörleri sorgular arasında değişmediği hâlde normların 1.000 kez
tekrar hesaplanması gereksiz iş ve yoğun bellek trafiği yaratıyor. Profilde
`np.linalg.norm` payının yaklaşık `%83,9` olması bu teşhisi doğrudan
destekliyor.

CPU kullanımı 10K-100K aralığında yaklaşık `%738-%742`, 250K'da `%640`,
500K'da `%657` ve 1M'de `%468` oldu. Thread değişkenleri ayarlanmamışken
NumPy/BLAS'ın dahili paralelliği kullandığı görülüyor. Veri büyüdükçe
CPU doluluğunun düşmesi, işin giderek cache/bellek bant genişliği tarafından
sınırlanmasıyla uyumludur. Bu bir çıkarımdır; donanım performance
counter'ı olmadan kesin cache-miss veya bandwidth oranı söylenemez.

## 8. Ölçüm sınırları

- Suite koşusunda 1M ortalama arama 204,790 ms, profiler koşusunda yaklaşık
  168 ms oldu. Ayrı koşular arasındaki bu fark arka plan yükü, termal/frekans
  durumu, koşu sırası ve profiling etkileriyle ilişkili olabilir. Resmî bir
  performans iddiası için aynı koşulda birden fazla tekrarın medyanı alınmalı.
- BLAS/OpenMP thread sayısı sabitlenmedi. Farklı makinelerle veya sürümlerle
  karşılaştırma yapılmadan önce thread sayısı açıkça belirlenmeli.
- İlk sorgu ayrı raporlandı ancak page cache temizlenmediği için gerçek bir
  cold-disk benchmark değildir.
- RSS yaklaşık ve platforma bağlıdır; peak RSS o ana kadarki process tepesidir.
- Yalnızca cosine, 128 dimension, k=10 ve metadata kapalı senaryo çalıştı.
  L2, dot, farklı dimension/k ve metadata için ayrı eğriler gerekir.
- Benchmark sonuçları geliştirme ağacından alındı. Bir release etiketi ve
  temiz worktree ile korunmuş baseline henüz yok.

## 9. Öncelikli optimizasyon planı

Teşhise göre mantıklı sıra şudur:

1. Cosine corpus normlarını ingestion/load sırasında bir kez hesaplayıp cache'le;
   add/delete/restore lifecycle'ında doğruluğunu koru.
2. Query başına allocation'ları azalt; özellikle denominator, valid ID ve skor
   ara dizilerini ölç.
3. Thread sayısı sabitlenmiş ve en az üç tekrarlı baseline/candidate suite'i
   çalıştır; doğruluğu her adayda oracle ile koru.
4. Norm cache sonrası profili tekrar al. Ancak yeni profilde anlamlı hâle gelirse
   top-k, tombstone filtreleme ve metadata/ID resolution yolunu optimize et.
5. Exact arama için batch API, block-wise scan veya native/Numba/SIMD seçeneklerini
   ayrı adaylar olarak ölç.
6. Milyonlarca vektörde düşük gecikme ana hedefse HNSW/IVF gibi ANN index'i
   exact baseline'dan ayrı bir backend olarak uygula; latency kazancını Recall@K
   ile birlikte raporla.

İlk optimizasyonun hedefi “Numba eklemek” gibi araç odaklı değil, profilin
gösterdiği tekrar corpus normu hesaplamasını ortadan kaldırmak olmalıdır.

## 10. Sonuç

Mevcut uygulama doğru sonuç üreten, snapshot persistence'ı olan ve 1M vektöre
kadar davranışı ölçülmüş bir exact baseline'dır. 1M'de yaklaşık 200 ms p50
ve 4,88 QPS, public release'i engelleyen gizli bir hata değil; exact scan ve
mevcut cosine implementasyonunun bugünkü maliyetidir. Buna karşın profil,
optimizasyon için çok net bir ilk hedef vermektedir: her sorguda corpus normlarını
yeniden hesaplamamak.
