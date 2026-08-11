/* Lung Cancer AI Warning System - Service Worker
   策略：网络优先、离线兜底。
   - 正常联网时：请求走网络，并把界面外壳（HTML/JS/CSS/图标）顺手缓存；
   - 断网时：返回缓存的外壳；
   - 推理/实时通道（/queue、/api、/gradio_api、/config）永不缓存，
     预测必须实时联网，避免把陈旧结果缓存下来。 */
const CACHE_NAME = 'lungai-shell-v1';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        cache.addAll([
          '/',
          '/manifest.json',
          '/pwa/icons/icon-192.png',
          '/pwa/icons/icon-512.png',
          '/pwa/icons/icon-192-maskable.png',
          '/pwa/icons/icon-512-maskable.png'
        ])
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

// 从导航响应里提取 Gradio 的静态资源地址，尽量在首次访问时把外壳资源补齐
function precacheAssetsFromHtml(html, cache) {
  if (!html) return Promise.resolve();
  const urls = [];
  const re =
    /(?:href|src)\s*=\s*["'](\/assets\/[^"']+\.(?:css|js))["']/g;
  let m = null;
  while ((m = re.exec(html)) !== null) {
    urls.push(m[1]);
  }
  return cache.addAll(urls).catch(() => {});
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // 推理 / 实时通信 / 配置：绝不缓存，也不离线兜底
  if (
    url.pathname.startsWith('/queue/') ||
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/gradio_api/') ||
    url.pathname === '/config'
  ) {
    return;
  }

  // 网络优先；失败时回退到缓存（导航请求兜底返回缓存的首页外壳）
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => {
              cache.put(req, copy);
              if (req.mode === 'navigate') {
                return res
                  .clone()
                  .text()
                  .then((html) => precacheAssetsFromHtml(html, cache));
              }
            })
            .catch(() => {});
        }
        return res;
      })
      .catch(() =>
        caches.match(req).then((hit) => {
          if (hit) return hit;
          if (req.mode === 'navigate') {
            return caches.match('/');
          }
          return Response.error();
        })
      )
  );
});
