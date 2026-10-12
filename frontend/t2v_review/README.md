# Evaluator Review

Read-only viewer for evaluator detail bundles, normally
`evaluation/<model>/<run>/t2v_debug/<sample_id>_<language>/`.

From the repository root:

```bash
REVIEW_DIR=./evaluation/model/run/t2v_debug \
./scripts/review/start_t2v.sh
```

Open `http://127.0.0.1:8767`. `REVIEW_DIR` may point to the entire review root
or one case folder; `PORT` and `HOST` control the listener. `REVIEW_DIR` is
required by the launcher.
Set `VIDEO_DIR` to a directory containing `<sample_id>.mp4` to enable video;
it may be incomplete, and sampling frames work without videos.

For VSCode/CloudML port forwarding, start with `HOST=0.0.0.0` and forward the
same `PORT` shown by the launcher. The dataset review UI uses a different port
(normally `8765`); its forwarded URL cannot be reused for this UI.

The API serves only files under the selected case folder and, when explicitly
enabled, matching video files under `VIDEO_DIR`. It sends a reduced config,
omits absolute video paths and VLM endpoints, and serves no arbitrary paths.
The raw bundle on disk still contains private paths and endpoints: do not
publish it directly. If exposed outside localhost, use an authenticated
reverse proxy and review prompts, input records, and images for sensitive data.

The interface supports Chinese and English from the language switcher in the
header. The selected language is kept in browser local storage. Click any
source or marker image to open the viewer; use the zoom controls or mouse wheel
to inspect details, drag while zoomed to pan, double-click to toggle zoom, and
use reset to return to the original view.
