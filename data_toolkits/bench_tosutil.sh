#!/bin/bash
# 测试 tosutil 下载 10 个 sha256 对象的速度

BUCKET="lhtest"
KEY_BASE="3diclight/3diclight_even_8w9/renders_3diclight_neural_graffer_0309/eevee/eevee"
DEST="/tmp/bench_tosutil"
TOSUTIL="/root/tosutil"
J=${1:-50}   # 第一个参数可以调整 -j，默认 50

OBJECTS=(
  9b7487e3eea2010a32355d8b92e02560f2e3d9e1b8ac006a74d17d74c5f072a5
  b5e9911c4732bdb342b638b2a0aa031c2e36524391a644e072ced1d145ca196e
  25a8494910c28824a7dcbc09b2b353fbc00172149bdce583c830f3bbb1f975d9
  f32e9c10a3cbbd781e8b383b9e44f77894a61749dd77164b9956d6a388062fcd
  006c621041d9863d29ad263fcd1fcecd02986d4a9c5782070981abbae9bb89cd
  7340ff708d8736e464e98df8d9933812e10efe2ab607fd6a8d80a9be00892c13
  37b1fa5fed95f8dacecc868446fa0a6a5605b39378903b9bf21fc09d9c71f66f
  b4d1d8cd76957129e1d7bfdeb30779fde7018770c8482251d0f430ef0493b147
  98fd0a583da39d198f66686858950b5142330dcdfcb4309e5eab16519e5546c5
  f55ad69b6ac2207ca91a3f1d0d5ecf9a48481f7be5cc41c28f425bc2a1044768
)

rm -rf "$DEST" && mkdir -p "$DEST"
echo "=== tosutil benchmark: -j=$J, 10 objects, sequential ==="
echo ""

TOTAL_BYTES=0
TOTAL_TIME=0

for sha in "${OBJECTS[@]}"; do
  SRC="tos://${BUCKET}/${KEY_BASE}/${sha}/"
  DEST_OBJ="$DEST/$sha"
  mkdir -p "$DEST_OBJ"

  T0=$(date +%s%N)
  "$TOSUTIL" cp "$SRC" "$DEST_OBJ/" -r -f -j=$J -p=1 > /dev/null 2>&1
  T1=$(date +%s%N)

  ELAPSED_MS=$(( (T1 - T0) / 1000000 ))
  BYTES=$(du -sb "$DEST_OBJ" 2>/dev/null | awk '{print $1}')
  FILES=$(find "$DEST_OBJ" -type f | wc -l)

  awk -v sha="${sha:0:16}" -v b="$BYTES" -v ms="$ELAPSED_MS" -v f="$FILES" \
    'BEGIN { mb=b/1048576; s=ms/1000; mbps=(s>0)?mb/s:0;
             printf "  %s...  %.0f MB / %d files  %.1fs  (%.0f MB/s)\n", sha, mb, f, s, mbps }'

  TOTAL_BYTES=$(( TOTAL_BYTES + BYTES ))
  TOTAL_TIME=$(( TOTAL_TIME + ELAPSED_MS ))
done

echo ""
awk -v b="$TOTAL_BYTES" -v ms="$TOTAL_TIME" \
  'BEGIN { mb=b/1048576; s=ms/1000; mbps=(s>0)?mb/s:0;
           printf "=== 合计: %.0f MB  %.1fs  平均 %.0f MB/s ===\n", mb, s, mbps }'

rm -rf "$DEST"
