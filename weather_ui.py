import html
import textwrap

from weather_services import (
    THUNDERSTORM_WEATHER_CODES,
    safe_display_number,
)


def weather_icon_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "?"
    if 200 <= code <= 232:
        return "!!"
    if 300 <= code <= 321:
        return ".."
    if 500 <= code <= 531:
        return "//"
    if 600 <= code <= 622:
        return "SN"
    if 700 <= code <= 781:
        return "~~"
    if code == 800:
        return "SUN"
    if 801 <= code <= 804:
        return "CL"
    if code in THUNDERSTORM_WEATHER_CODES:
        return "!!"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "//"
    if code in {51, 53, 55, 56, 57}:
        return ".."
    if code in {45, 48}:
        return "~~"
    if code in {2, 3}:
        return "CL"
    if code in {0, 1}:
        return "SUN"
    return "WX"


def weather_theme_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "neutral"
    if code in THUNDERSTORM_WEATHER_CODES or 200 <= code <= 232:
        return "storm"
    if code in {61, 63, 65, 66, 67, 80, 81, 82} or 500 <= code <= 531:
        return "rain"
    if code in {45, 48} or 700 <= code <= 781:
        return "mist"
    if code in {2, 3} or 801 <= code <= 804:
        return "cloud"
    if code in {0, 1, 800}:
        return "clear"
    return "neutral"


def weather_symbol_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "â˜"
    if 200 <= code <= 232 or code in THUNDERSTORM_WEATHER_CODES:
        return "âš¡"
    if 300 <= code <= 321:
        return "â˜‚"
    if 500 <= code <= 531:
        return "â˜”"
    if 600 <= code <= 622:
        return "â„"
    if 700 <= code <= 781:
        return "â‰‹"
    if code == 800 or code in {0, 1}:
        return "â˜€"
    if code in {801, 2}:
        return "â›…"
    return "â˜"


def build_weather_trend_svg(items):
    usable = []
    for item in items:
        try:
            temp = float(item.get("temperature_c"))
        except (TypeError, ValueError):
            continue
        try:
            rain = float(item.get("precip_mm") or 0.0)
        except (TypeError, ValueError):
            rain = 0.0
        usable.append((temp, rain, str(item.get("time", "-"))))

    if len(usable) < 2:
        return ""

    width = 520
    height = 110
    x_start = 18
    x_step = (width - 36) / max(len(usable) - 1, 1)
    temps = [item[0] for item in usable]
    min_temp = min(temps)
    max_temp = max(temps)
    temp_span = max(max_temp - min_temp, 1.0)
    points = []
    labels = []
    rain_bars = []

    for idx, (temp, rain, time_label) in enumerate(usable):
        x = x_start + idx * x_step
        y = 68 - ((temp - min_temp) / temp_span) * 36
        points.append(f"{x:.1f},{y:.1f}")
        labels.append((x, y, temp, time_label))
        bar_height = min(32.0, max(3.0, rain * 9.0)) if rain > 0 else 3.0
        rain_bars.append(
            f"<rect x='{x - 5:.1f}' y='{82 - bar_height:.1f}' width='10' height='{bar_height:.1f}' rx='4' />"
        )

    grid_lines = "".join(
        f"<line x1='{x_start + idx * x_step:.1f}' y1='18' x2='{x_start + idx * x_step:.1f}' y2='92' />"
        for idx in range(len(usable))
    )
    circles = "".join(
        f"<circle cx='{point.split(',')[0]}' cy='{point.split(',')[1]}' r='3' />"
        for point in points
    )
    text_labels = "".join(
        f"<text x='{x:.1f}' y='{max(12.0, y - 8):.1f}' text-anchor='middle'>{temp:.0f}°</text>"
        for x, y, temp, _ in labels
    )
    time_labels = "".join(
        f"<text x='{x:.1f}' y='104' text-anchor='middle'>{time_label}</text>"
        for x, _, _, time_label in labels
    )
    return (
        "<svg class='weather-trend-svg' viewBox='0 0 520 110' role='img' aria-label='Grafik tren suhu dan hujan'>"
        f"<g class='weather-trend-grid'>{grid_lines}</g>"
        f"<g class='weather-trend-rain'>{''.join(rain_bars)}</g>"
        f"<polyline class='weather-trend-line' points='{' '.join(points)}' />"
        f"<g class='weather-trend-points'>{circles}</g>"
        f"<g class='weather-trend-labels'>{text_labels}</g>"
        f"<g class='weather-trend-axis'>{time_labels}</g>"
        "</svg>"
    )


def build_weather_trend_svg(items):
    usable = []
    for item in items:
        try:
            temp = float(item.get("temperature_c"))
        except (TypeError, ValueError):
            continue
        try:
            rain = float(item.get("precip_mm") or 0.0)
        except (TypeError, ValueError):
            rain = 0.0
        usable.append((temp, rain, str(item.get("time", "-"))))

    if len(usable) < 2:
        return ""

    temps = [item[0] for item in usable]
    min_temp = min(temps)
    max_temp = max(temps)
    temp_span = max(max_temp - min_temp, 1.0)
    slots = []
    for temp, rain, time_label in usable:
        level = (temp - min_temp) / temp_span
        try:
            rain_level = min(1.0, max(0.0, float(rain) / 10.0))
        except (TypeError, ValueError):
            rain_level = 0.0
        slots.append(
            "<div class='trend-slot' "
            f"style='--level:{level:.3f}; --rain-level:{rain_level:.3f};'>"
            f"<span class='trend-value'>{temp:.0f}°</span>"
            "<div class='trend-stage'><i></i><b></b><em></em></div>"
            f"<small>{html.escape(time_label)}</small>"
            "</div>"
        )
    return (
        "<div class='weather-trend-chart' role='img' aria-label='Grafik tren suhu per jam'>"
        + "".join(slots)
        + "</div>"
    )


def build_rain_chance_bars(items):
    bars = ""
    max_pop = 1
    for item in items:
        try:
            max_pop = max(max_pop, int(item.get("pop_pct") or 0))
        except (TypeError, ValueError):
            pass
    for item in items:
        try:
            pop_num = int(item.get("pop_pct") or 0)
        except (TypeError, ValueError):
            pop_num = 0
        bar_height = 12 + (pop_num / max_pop) * 58
        bars += (
            "<div class='rain-bar-wrap'>"
            f"<i style='height:{bar_height:.0f}px'></i>"
            f"<span>{pop_num}%</span>"
            f"<small>{item.get('time', '-')}</small>"
            "</div>"
        )
    return bars


def build_weather_trend_svg(items):
    usable = []
    for item in items:
        try:
            temp = float(item.get("temperature_c"))
        except (TypeError, ValueError):
            continue
        usable.append((temp, str(item.get("time", "-"))))

    if len(usable) < 2:
        return ""

    temps = [item[0] for item in usable]
    min_temp = min(temps)
    max_temp = max(temps)
    temp_span = max(max_temp - min_temp, 1.0)
    levels = [(temp - min_temp) / temp_span for temp, _ in usable]
    n_items = len(usable)
    slots = []
    for idx, (temp, time_label) in enumerate(usable):
        level = levels[idx]
        slots.append(
            "<div class='trend-slot' "
            f"style='--level:{level:.3f};'>"
            f"<span class='trend-value'>{temp:.0f}°</span>"
            "<div class='trend-stage'><i></i><b></b></div>"
            f"<small>{html.escape(time_label)}</small>"
            "</div>"
        )
    return (
        "<div class='weather-trend-chart' role='img' aria-label='Grafik tren suhu per jam'>"
        + "".join(slots)
        + "</div>"
    )


def build_rain_chance_chart(items):
    usable = []
    for item in items:
        try:
            pop_num = int(item.get("pop_pct") or 0)
        except (TypeError, ValueError):
            pop_num = 0
        usable.append((pop_num, str(item.get("time", "-"))))

    if not usable:
        return ""

    max_pop = max(1, max(pop for pop, _ in usable))
    levels = [pop / max_pop for pop, _ in usable]
    slots = []
    for idx, (pop_num, time_label) in enumerate(usable):
        level = levels[idx]
        slots.append(
            "<div class='rain-bar-wrap' "
            f"style='--rain-level:{level:.3f};'>"
            "<i></i>"
            f"<span>{pop_num}%</span>"
            f"<small>{html.escape(time_label)}</small>"
            "</div>"
        )
    return "".join(slots)


def weather_symbol_for_code(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "&#9729;"
    if 200 <= code <= 232 or code in THUNDERSTORM_WEATHER_CODES:
        return "&#9889;"
    if 300 <= code <= 321:
        return "&#9748;"
    if 500 <= code <= 531:
        return "&#9748;"
    if 600 <= code <= 622:
        return "&#10052;"
    if 700 <= code <= 781:
        return "&#8779;"
    if code == 800 or code in {0, 1}:
        return "&#9728;"
    if code in {801, 2}:
        return "&#9925;"
    return "&#9729;"


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        thunder_text = row.get("Last Thunderstorm Indication", "-")
        storm_class = "weather-storm-muted"
        if row.get("Last Thunderstorm Time"):
            thunder_text = f"{row.get('Last Thunderstorm Time')} | {row.get('Last Thunderstorm Weather', '-')}"
            storm_class = "weather-storm-active"
        theme = weather_theme_for_code(row.get("Weather Code"))
        icon_url = row.get("Weather Icon URL")
        if icon_url:
            visual_html = f"<img class='weather-hero-img' src='{icon_url}' alt='{row.get('Current Weather', 'Weather')}' />"
        else:
            visual_html = f"<div class='weather-hero-fallback'>{weather_icon_for_code(row.get('Weather Code'))}</div>"
        visual_caption = str(row.get("Current Weather") or "-").title()
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])
        forecast_items_html = ""
        temp_values = []
        precip_values = []
        for item in forecast_items:
            temp_value = item.get("temperature_c")
            precip_value = item.get("precip_mm") or 0.0
            try:
                temp_values.append(float(temp_value))
            except (TypeError, ValueError):
                pass
            try:
                precip_values.append(float(precip_value))
            except (TypeError, ValueError):
                precip_value = 0.0
            item_icon = (
                f"<img src='{item.get('icon_url')}' alt='{item.get('description', 'Forecast')}' />"
                if item.get("icon_url")
                else f"<strong>{weather_icon_for_code(item.get('weather_code'))}</strong>"
            )
            pop_label = f"{item.get('pop_pct')}%" if item.get("pop_pct") is not None else "-"
            temp_label = safe_display_number(temp_value, 0, "°C")
            precip_label = safe_display_number(precip_value, 1, " mm") if precip_value else ""
            risk_value = item.get("pop_pct") if item.get("pop_pct") is not None else 0
            precip_height = min(28, max(3, float(precip_value) * 8.0)) if precip_value else 3
            forecast_items_html += (
                "<div class='weather-forecast-item'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{temp_label}</b>"
                f"<em>{item.get('description', '-')}</em>"
                f"<small>{pop_label}</small>"
                "<div class='weather-precip-track'>"
                f"<i style='height:{precip_height}px'></i>"
                "</div>"
                f"<label>{precip_label}</label>"
                f"<mark>{risk_value}</mark>"
                "</div>"
            )
        temp_min = min(temp_values) if temp_values else None
        temp_max = max(temp_values) if temp_values else None
        temp_range = (
            f"{safe_display_number(temp_min, 0, ' C')} - {safe_display_number(temp_max, 0, ' C')}"
            if temp_min is not None and temp_max is not None
            else "-"
        )
        forecast_summary = forecast.get("summary", "Forecast belum tersedia.")
        forecast_pop = (
            f"{forecast.get('max_pop') * 100.0:.0f}%"
            if forecast.get("max_pop") is not None
            else "-"
        )
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        trend_svg = build_weather_trend_svg(forecast_items)
        visibility_km = None
        try:
            visibility_km = float(row.get("Visibility m")) / 1000.0
        except (TypeError, ValueError):
            visibility_km = None
        cards_html.append(
            textwrap.dedent(f"""
            <div class="weather-card weather-theme-{theme}">
                <div class="weather-effect weather-effect-{theme}"></div>
                <div class="weather-glass">
                    <div class="weather-top">
                        <div class="weather-now">
                            <div class="weather-visual-badge">{visual_html}</div>
                            <div>
                                <div class="weather-desc">{row.get('Current Weather', '-')}</div>
                                <div class="weather-location">{row.get('Location', '-')}</div>
                                <div class="weather-tower">{row.get('Tower', '-')}</div>
                            </div>
                        </div>
                        <div class="weather-temp-block">
                            <div class="weather-temp">{safe_display_number(row.get('Temperature C'), 1, ' C')}</div>
                            <div class="weather-range">{temp_range}</div>
                        </div>
                    </div>

                    <div class="weather-attribute-list">
                        <div><span>Jarak pandang</span><strong>{safe_display_number(visibility_km, 1, ' km')}</strong></div>
                        <div><span>Terasa seperti</span><strong>{safe_display_number(row.get('Feels Like C'), 1, ' C')}</strong></div>
                        <div><span>Kelembapan</span><strong>{safe_display_number(row.get('Humidity %'), 0, '%')}</strong></div>
                        <div><span>Angin</span><strong>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</strong></div>
                        <div><span>Hujan saat ini</span><strong>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</strong></div>
                        <div><span>Tutupan awan</span><strong>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</strong></div>
                    </div>

                    <div class="weather-forecast-row">
                        <div class="weather-forecast-copy">
                            <span>Prakiraan 12 jam</span>
                            <strong>{forecast_summary}</strong>
                            <small>{forecast_signal} | Peluang hujan tertinggi {forecast_pop} | Hujan total {forecast_precip}</small>
                        </div>
                        {trend_svg}
                    </div>

                    <div class="weather-forecast-strip">
                        {forecast_items_html}
                    </div>

                    <div class="weather-storm {storm_class}">
                        <span>Indikasi thunderstorm</span>
                        <strong>{thunder_text}</strong>
                    </div>

                    <div class="weather-footer">
                        <span>Kumulatif {safe_display_number(row.get('Cumulative km'), 3, ' km')}</span>
                        <span>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)}</span>
                        <span>Update {row.get('Weather Time') or '-'} | {row.get('Weather Source', '-')}</span>
                    </div>
                </div>
            </div>
            """)
        )
    return (
        textwrap.dedent("""
        <style>
        .weather-card-wrap {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 14px;
            margin: 12px 0 14px 0;
        }
        .weather-card {
            border: 1px solid rgba(148, 163, 184, 0.45);
            border-radius: 8px;
            padding: 14px 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
            color: #0f172a;
            break-inside: avoid;
            page-break-inside: avoid;
            overflow: hidden;
            position: relative;
        }
        .weather-layout {
            display: grid;
            grid-template-columns: minmax(360px, 0.9fr) minmax(440px, 1.1fr);
            gap: 12px;
            align-items: stretch;
        }
        .weather-current-panel,
        .weather-forecast-panel {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: rgba(255, 255, 255, 0.52);
            padding: 12px;
            height: 100%;
        }
        .weather-current-panel {
            display: flex;
            flex-direction: column;
        }
        .weather-forecast-panel {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .weather-theme-clear {
            background: linear-gradient(135deg, #fff7ed 0%, #dbeafe 100%);
        }
        .weather-theme-cloud {
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 56%, #e2e8f0 100%);
        }
        .weather-theme-rain {
            background: linear-gradient(135deg, #eef6ff 0%, #c7d2fe 100%);
        }
        .weather-theme-storm {
            background: linear-gradient(135deg, #eef2ff 0%, #fed7aa 100%);
        }
        .weather-theme-mist {
            background: linear-gradient(135deg, #f8fafc 0%, #e5e7eb 100%);
        }
        .weather-hero,
        .weather-meta {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-start;
        }
        .weather-hero {
            min-height: 112px;
            align-items: center;
            padding: 0;
            border-bottom: 0;
        }
        .weather-copy {
            min-width: 0;
            flex: 1;
        }
        .weather-role {
            color: #ef4444;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 3px;
        }
        .weather-tower {
            font-size: 14px;
            font-weight: 700;
            line-height: 1.25;
        }
        .weather-temp {
            font-size: 42px;
            line-height: 1;
            font-weight: 800;
            margin-top: 18px;
        }
        .weather-desc {
            color: #334155;
            font-size: 16px;
            margin-top: 4px;
            text-transform: capitalize;
        }
        .weather-visual-card {
            width: 154px;
            border-radius: 8px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(255, 255, 255, 0.78);
            box-shadow: 0 18px 45px rgba(15,23,42,0.13);
            flex: 0 0 auto;
        }
        .weather-visual {
            width: 100%;
            height: 110px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background:
                radial-gradient(circle at 50% 38%, rgba(255,255,255,0.94), rgba(255,255,255,0.42) 54%, rgba(226,232,240,0.42));
        }
        .weather-hero-img {
            width: 104px;
            height: 104px;
            object-fit: contain;
            filter: drop-shadow(0 10px 12px rgba(15, 23, 42, 0.18));
        }
        .weather-visual-caption {
            margin-top: 5px;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
        }
        .weather-hero-fallback {
            width: 96px;
            height: 96px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #0f172a;
            color: #ffffff;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 0;
        }
        .weather-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .weather-grid div {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(203, 213, 225, 0.75);
            border-radius: 8px;
            padding: 8px 9px;
        }
        .weather-grid span,
        .weather-storm span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 1px;
        }
        .weather-grid strong,
        .weather-storm strong {
            font-size: 13px;
        }
        .weather-storm {
            margin-top: 8px;
            border-radius: 8px;
            padding: 8px 10px;
            border: 1px solid rgba(203, 213, 225, 0.8);
            background: rgba(255, 255, 255, 0.72);
        }
        .weather-forecast-head {
            margin-bottom: 8px;
        }
        .weather-forecast-head span,
        .weather-forecast-metrics span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .weather-forecast-head strong {
            font-size: 14px;
        }
        .weather-forecast-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 7px;
            margin-bottom: 8px;
        }
        .weather-forecast-metrics div {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.75);
            background: rgba(248, 250, 252, 0.85);
            padding: 7px 8px;
        }
        .weather-alert-pill {
            margin: 0 0 8px 0;
            padding: 7px 9px;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.32);
            color: #0f172a;
            font-size: 12px;
            font-weight: 700;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 8px;
            margin-top: 0;
            overflow-x: auto;
            padding-bottom: 2px;
            flex: 1;
        }
        .weather-forecast-item {
            flex: 0 0 86px;
            min-width: 86px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(241,245,249,0.78));
            padding: 7px 6px 6px 6px;
            text-align: center;
            display: grid;
            grid-template-rows: auto 30px auto auto auto 30px;
            align-content: center;
            gap: 1px;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            display: block;
            color: #475569;
            font-size: 10px;
        }
        .weather-forecast-item em {
            display: block;
            color: #0f172a;
            font-size: 10px;
            font-style: normal;
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item b {
            display: block;
            color: #0f172a;
            font-size: 13px;
            line-height: 1;
        }
        .weather-forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            margin: -3px auto -3px auto;
        }
        .weather-forecast-item strong {
            display: grid;
            place-items: center;
            width: 28px;
            height: 28px;
            margin: 0 auto;
            border-radius: 50%;
            background: #0f172a;
            color: #ffffff;
            font-size: 9px;
        }
        .weather-precip-track {
            width: 16px;
            height: 30px;
            margin: 1px auto 0 auto;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            border-radius: 999px;
            background: rgba(203, 213, 225, 0.45);
            overflow: hidden;
        }
        .weather-precip-track i {
            display: block;
            width: 100%;
            min-height: 3px;
            border-radius: 999px;
            background: linear-gradient(180deg, #60a5fa, #2563eb);
        }
        .weather-storm-active {
            border-color: rgba(234, 88, 12, 0.55);
            background: rgba(255, 237, 213, 0.9);
        }
        .weather-storm-muted strong {
            color: #475569;
        }
        .weather-meta,
        .weather-time {
            color: #64748b;
            font-size: 11px;
            margin-top: 8px;
        }
        @media print {
            .weather-card-wrap {
                grid-template-columns: 1fr;
            }
            .weather-card {
                box-shadow: none;
            }
        }
        @media (max-width: 980px) {
            .weather-layout {
                grid-template-columns: 1fr;
                align-items: start;
            }
            .weather-current-panel,
            .weather-forecast-panel {
                height: auto;
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                padding: 10px;
            }
            .weather-current-panel,
            .weather-forecast-panel {
                padding: 10px;
                min-width: 0;
            }
            .weather-hero {
                align-items: flex-start;
                min-height: 0;
                gap: 10px;
            }
            .weather-visual-card {
                width: 92px;
                padding: 8px;
            }
            .weather-visual {
                height: 76px;
            }
            .weather-hero-img {
                width: 72px;
                height: 72px;
            }
            .weather-visual-caption {
                font-size: 10px;
            }
            .weather-temp {
                font-size: 34px;
            }
            .weather-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .weather-forecast-head {
                grid-template-columns: 1fr;
            }
            .weather-forecast-metrics {
                grid-template-columns: repeat(3, minmax(96px, 1fr));
                overflow-x: auto;
                padding-bottom: 2px;
            }
            .weather-forecast-metrics div {
                min-width: 96px;
            }
            .weather-forecast-strip {
                display: flex;
                gap: 7px;
                overflow-x: auto;
                max-width: 100%;
                flex: 0 0 auto;
            }
            .weather-forecast-item {
                flex: 0 0 78px;
                min-width: 78px;
            }
            .weather-meta {
                display: grid;
                grid-template-columns: 1fr;
                gap: 2px;
            }
            .weather-time {
                overflow-wrap: anywhere;
            }
        }
        .weather-card {
            border-radius: 16px;
            padding: 0;
            color: #f8fafc;
            background:
                radial-gradient(circle at 66% -14%, rgba(255, 239, 163, 0.95), rgba(255, 239, 163, 0.05) 15%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(239, 68, 68, 0.72), transparent 32%),
                linear-gradient(135deg, #0f7caf 0%, #0f5d82 46%, #153d5a 100%);
            border: 1px solid rgba(255, 255, 255, 0.22);
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.22);
            overflow: hidden;
            position: relative;
        }
        .weather-theme-cloud,
        .weather-theme-mist {
            background:
                radial-gradient(circle at 66% -14%, rgba(255, 239, 163, 0.78), rgba(255, 239, 163, 0.04) 14%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(239, 68, 68, 0.62), transparent 32%),
                linear-gradient(135deg, #1675a7 0%, #195779 48%, #213d56 100%);
        }
        .weather-theme-rain,
        .weather-theme-storm {
            background:
                radial-gradient(circle at 64% -16%, rgba(250, 204, 21, 0.52), rgba(250, 204, 21, 0.03) 14%, transparent 25%),
                radial-gradient(circle at 0% 100%, rgba(220, 38, 38, 0.66), transparent 32%),
                linear-gradient(135deg, #0f5276 0%, #173b5a 48%, #202b44 100%);
        }
        .weather-effect {
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0.28;
            background-image:
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 58px 58px;
            mask-image: linear-gradient(180deg, transparent, #000 18%, #000 74%, transparent);
        }
        .weather-glass {
            position: relative;
            z-index: 1;
            padding: 22px 28px 20px 28px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.12);
            backdrop-filter: blur(6px);
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 18px;
            align-items: start;
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 18px;
            min-width: 0;
        }
        .weather-visual-badge {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            flex: 0 0 auto;
            background: rgba(255, 255, 255, 0.10);
        }
        .weather-visual-badge .weather-hero-img {
            width: 82px;
            height: 82px;
            object-fit: contain;
            filter: drop-shadow(0 12px 16px rgba(0,0,0,0.25));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 58px;
            height: 58px;
            font-size: 13px;
            background: rgba(255, 255, 255, 0.16);
            color: #ffffff;
        }
        .weather-desc {
            color: #ffffff;
            font-size: 30px;
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
            text-transform: capitalize;
        }
        .weather-location {
            color: rgba(255, 255, 255, 0.68);
            font-size: 14px;
            font-weight: 700;
            margin-top: 2px;
        }
        .weather-tower {
            color: rgba(255, 255, 255, 0.72);
            font-size: 12px;
            margin-top: 5px;
        }
        .weather-temp-block {
            text-align: right;
        }
        .weather-temp {
            color: #ffffff;
            font-size: 34px;
            line-height: 1;
            font-weight: 800;
            margin: 0;
        }
        .weather-range {
            color: rgba(255, 255, 255, 0.58);
            font-size: 13px;
            margin-top: 5px;
            font-weight: 700;
        }
        .weather-attribute-list {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px 24px;
            margin-top: 20px;
        }
        .weather-attribute-list div {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: rgba(255, 255, 255, 0.78);
            font-size: 13px;
            font-weight: 650;
        }
        .weather-attribute-list strong {
            color: #ffffff;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.1fr);
            gap: 16px;
            align-items: center;
            margin-top: 18px;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            display: block;
            color: rgba(255, 255, 255, 0.66);
            font-size: 12px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #ffffff;
            font-size: 15px;
            margin: 5px 0 6px 0;
        }
        .weather-trend-svg {
            width: 100%;
            height: 110px;
            display: block;
            overflow: visible;
        }
        .weather-trend-grid line {
            stroke: rgba(255, 255, 255, 0.18);
            stroke-dasharray: 4 5;
        }
        .weather-trend-rain rect {
            fill: rgba(96, 165, 250, 0.75);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #111827;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 14px;
            overflow-x: auto;
            padding: 12px 2px 8px 2px;
            margin-top: 2px;
        }
        .weather-forecast-item {
            flex: 0 0 74px;
            min-width: 74px;
            border: 0;
            border-radius: 0;
            background: transparent;
            color: rgba(255, 255, 255, 0.74);
            padding: 0;
            display: grid;
            grid-template-rows: 18px 34px 18px 16px 16px 28px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            color: rgba(255, 255, 255, 0.72);
            font-size: 11px;
        }
        .weather-forecast-item img {
            width: 34px;
            height: 34px;
            object-fit: contain;
            filter: drop-shadow(0 8px 10px rgba(0,0,0,0.25));
            margin: 0;
        }
        .weather-forecast-item b {
            color: #ffffff;
            font-size: 13px;
        }
        .weather-forecast-item em {
            max-width: 70px;
            color: rgba(255, 255, 255, 0.62);
            font-size: 10px;
        }
        .weather-forecast-item strong {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }
        .weather-precip-track {
            width: 22px;
            height: 28px;
            background: rgba(255, 255, 255, 0.14);
        }
        .weather-precip-track i {
            background: linear-gradient(180deg, #7dd3fc, #2563eb);
        }
        .weather-storm {
            margin-top: 12px;
            border-radius: 999px;
            padding: 8px 12px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(15, 23, 42, 0.16);
        }
        .weather-storm span {
            color: rgba(255, 255, 255, 0.58);
            font-size: 11px;
        }
        .weather-storm strong {
            color: #ffffff;
            font-size: 12px;
        }
        .weather-storm-active {
            border-color: rgba(251, 191, 36, 0.72);
            background: rgba(251, 146, 60, 0.18);
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 12px;
            color: rgba(255, 255, 255, 0.54);
            font-size: 11px;
            flex-wrap: wrap;
        }
        @media (max-width: 900px) {
            .weather-glass {
                padding: 18px;
            }
            .weather-top,
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
                gap: 7px;
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                border-radius: 10px;
            }
            .weather-glass {
                padding: 14px;
                border-radius: 10px;
            }
            .weather-now {
                gap: 12px;
            }
            .weather-visual-badge {
                width: 54px;
                height: 54px;
            }
            .weather-visual-badge .weather-hero-img {
                width: 64px;
                height: 64px;
            }
            .weather-desc {
                font-size: 22px;
            }
            .weather-temp {
                font-size: 30px;
            }
            .weather-forecast-strip {
                gap: 10px;
            }
            .weather-forecast-item {
                flex-basis: 66px;
                min-width: 66px;
            }
        }
        /* Home Assistant-style compact forecast card override */
        .weather-card-wrap {
            place-items: start center;
            margin: 10px 0 12px 0;
        }
        .weather-card {
            width: min(100%, 760px);
            border-radius: 14px;
            background:
                radial-gradient(circle at 70% -20%, rgba(255, 236, 148, 0.86), rgba(255, 236, 148, 0.05) 16%, transparent 28%),
                radial-gradient(circle at -5% 115%, rgba(239, 68, 68, 0.62), transparent 30%),
                linear-gradient(135deg, #1277a6 0%, #145875 50%, #163f58 100%);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.20);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .weather-glass {
            padding: 18px 20px 16px 20px;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 14px;
        }
        .weather-top {
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 14px;
        }
        .weather-now {
            gap: 14px;
        }
        .weather-visual-badge {
            width: 58px;
            height: 58px;
            background: rgba(255, 255, 255, 0.11);
        }
        .weather-visual-badge .weather-hero-img {
            width: 68px;
            height: 68px;
        }
        .weather-desc {
            font-size: 23px;
            letter-spacing: 0;
        }
        .weather-location {
            font-size: 12px;
            margin-top: 3px;
        }
        .weather-tower {
            font-size: 11px;
            margin-top: 2px;
        }
        .weather-temp {
            font-size: 29px;
        }
        .weather-range {
            font-size: 11px;
        }
        .weather-attribute-list {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 7px 22px;
            margin-top: 16px;
        }
        .weather-attribute-list div {
            font-size: 12px;
        }
        .weather-forecast-row {
            grid-template-columns: minmax(0, 0.7fr) minmax(280px, 1fr);
            gap: 14px;
            margin-top: 16px;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            font-size: 11px;
        }
        .weather-forecast-copy strong {
            font-size: 13px;
            margin: 3px 0 5px 0;
        }
        .weather-trend-svg {
            height: 76px;
        }
        .weather-forecast-strip {
            gap: 12px;
            padding: 8px 0 6px 0;
            margin-top: 4px;
            scrollbar-width: thin;
        }
        .weather-forecast-item {
            flex-basis: 56px;
            min-width: 56px;
            grid-template-rows: 16px 28px 16px 14px 14px 22px;
        }
        .weather-forecast-item span,
        .weather-forecast-item small,
        .weather-forecast-item em {
            font-size: 9px;
        }
        .weather-forecast-item img {
            width: 28px;
            height: 28px;
        }
        .weather-forecast-item b {
            font-size: 11px;
        }
        .weather-precip-track {
            width: 18px;
            height: 22px;
        }
        .weather-storm {
            margin-top: 8px;
            padding: 7px 10px;
        }
        .weather-footer {
            margin-top: 8px;
            font-size: 10px;
        }
        @media (max-width: 820px) {
            .weather-card {
                width: 100%;
            }
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-trend-svg {
                height: 70px;
            }
        }
        @media (max-width: 520px) {
            .weather-glass {
                padding: 14px;
            }
            .weather-top {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
            }
            .weather-desc {
                font-size: 20px;
            }
            .weather-forecast-item {
                flex-basis: 54px;
                min-width: 54px;
            }
        }
        /* Close visual match to troinine/ha-weather-forecast-card demo */
        .weather-card-wrap {
            place-items: start center;
            margin: 10px 0;
        }
        .weather-card {
            width: min(100%, 552px);
            min-height: 428px;
            border-radius: 14px;
            color: #ffffff;
            background:
                radial-gradient(circle at 64% -8%, rgba(255, 248, 178, 0.96) 0 7%, rgba(255, 248, 178, 0.20) 10%, transparent 19%),
                radial-gradient(circle at -8% 105%, rgba(239, 68, 68, 0.82), rgba(239, 68, 68, 0.34) 18%, transparent 38%),
                linear-gradient(135deg, rgba(15, 132, 184, 0.96), rgba(20, 83, 115, 0.94) 52%, rgba(28, 65, 91, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24);
        }
        .weather-effect {
            opacity: 0.18;
            background-image:
                linear-gradient(90deg, rgba(255,255,255,0.07) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px);
            background-size: 44px 44px;
        }
        .weather-glass {
            min-height: 428px;
            padding: 26px 34px 26px 34px;
            background: rgba(15, 23, 42, 0.06);
            border-radius: 14px;
            display: grid;
            grid-template-rows: auto auto 1fr auto auto;
            gap: 0;
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: start;
            gap: 16px;
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .weather-visual-badge {
            width: 54px;
            height: 54px;
            background: transparent;
            border-radius: 50%;
        }
        .weather-visual-badge .weather-hero-img {
            width: 64px;
            height: 64px;
            filter: brightness(1.2) drop-shadow(0 8px 12px rgba(0,0,0,0.18));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 52px;
            height: 52px;
            background: #facc15;
            color: #1f2937;
        }
        .weather-desc {
            font-size: 30px;
            line-height: 1;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }
        .weather-location {
            font-size: 14px;
            color: rgba(255, 255, 255, 0.66);
            font-weight: 700;
            margin-top: 4px;
        }
        .weather-tower {
            display: none;
        }
        .weather-temp {
            font-size: 30px;
            line-height: 1;
            color: #ffffff;
            font-weight: 800;
            margin: 0;
        }
        .weather-range {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.48);
            font-weight: 800;
            margin-top: 8px;
        }
        .weather-attribute-list {
            margin-top: 28px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 46px;
        }
        .weather-attribute-list div {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            color: rgba(255, 255, 255, 0.76);
            font-size: 13px;
            font-weight: 800;
        }
        .weather-attribute-list strong {
            color: #ffffff;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: 0.92fr 1.08fr;
            gap: 18px;
            align-items: end;
            margin-top: 22px;
        }
        .weather-forecast-copy span {
            color: rgba(255,255,255,0.56);
            font-size: 12px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #ffffff;
            font-size: 13px;
            line-height: 1.25;
            margin-top: 6px;
        }
        .weather-forecast-copy small {
            display: block;
            color: rgba(255,255,255,0.58);
            font-size: 10px;
            margin-top: 6px;
        }
        .weather-trend-svg {
            width: 100%;
            height: 112px;
        }
        .weather-trend-grid line {
            stroke: rgba(255,255,255,0.18);
            stroke-dasharray: 5 5;
        }
        .weather-trend-rain rect {
            fill: rgba(96, 165, 250, 0.62);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #11314a;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 0 0 6px 0;
            margin-top: -106px;
            width: 100%;
            scrollbar-width: none;
        }
        .weather-forecast-strip::-webkit-scrollbar {
            display: none;
        }
        .weather-forecast-item {
            flex: 0 0 50px;
            min-width: 50px;
            background: transparent;
            border: 0;
            color: #ffffff;
            padding: 0;
            display: grid;
            grid-template-rows: 18px 26px 18px 13px 13px 28px 16px 24px;
            justify-items: center;
            text-align: center;
            gap: 0;
        }
        .weather-forecast-item span {
            color: rgba(255,255,255,0.54);
            font-size: 12px;
            font-weight: 800;
        }
        .weather-forecast-item img {
            width: 28px;
            height: 28px;
            object-fit: contain;
            filter: brightness(1.2) drop-shadow(0 6px 8px rgba(0,0,0,0.18));
        }
        .weather-forecast-item b {
            color: #ffffff;
            font-size: 12px;
            line-height: 1;
        }
        .weather-forecast-item em {
            max-width: 48px;
            color: rgba(255,255,255,0.48);
            font-size: 9px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item small {
            color: rgba(255,255,255,0.60);
            font-size: 10px;
        }
        .weather-precip-track {
            width: 18px;
            height: 28px;
            border-radius: 10px;
            background: rgba(255,255,255,0.20);
            display: flex;
            align-items: flex-end;
            justify-content: center;
            overflow: hidden;
        }
        .weather-precip-track i {
            width: 100%;
            min-height: 3px;
            background: linear-gradient(180deg, #93c5fd, #3b82f6);
        }
        .weather-forecast-item label {
            color: rgba(255,255,255,0.76);
            font-size: 10px;
            font-weight: 800;
            min-height: 14px;
        }
        .weather-forecast-item mark {
            display: grid;
            place-items: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            color: #ffffff;
            font-size: 11px;
            font-weight: 800;
            background: rgba(15,23,42,0.38);
            border: 3px solid #f59e0b;
            padding: 0;
        }
        .weather-storm {
            margin-top: 8px;
            border-radius: 999px;
            padding: 8px 12px;
            background: rgba(15,23,42,0.20);
            border: 1px solid rgba(245,158,11,0.80);
        }
        .weather-storm span {
            color: rgba(255,255,255,0.54);
            font-size: 10px;
        }
        .weather-storm strong {
            display: block;
            color: #ffffff;
            font-size: 11px;
            margin-top: 2px;
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-top: 8px;
            color: rgba(255,255,255,0.46);
            font-size: 10px;
            flex-wrap: wrap;
        }
        @media (max-width: 640px) {
            .weather-card {
                width: 100%;
                min-height: 0;
            }
            .weather-glass {
                min-height: 0;
                padding: 16px;
            }
            .weather-top,
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: 1fr;
                gap: 8px;
                margin-top: 18px;
            }
            .weather-forecast-strip {
                margin-top: 10px;
            }
            .weather-trend-svg {
                height: 86px;
            }
        }
        /* Transmission fault report weather card */
        .weather-card-wrap {
            place-items: stretch;
            margin: 10px 0 12px 0;
        }
        .weather-card {
            width: 100%;
            min-height: 0;
            border-radius: 10px;
            color: #0f172a;
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 100%);
            border: 1px solid rgba(148, 163, 184, 0.46);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.11);
        }
        .weather-effect {
            display: none;
        }
        .weather-glass {
            min-height: 0;
            padding: 16px;
            border-radius: 10px;
            background: transparent;
            display: grid;
            grid-template-rows: auto;
        }
        .weather-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 16px;
            align-items: start;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.32);
        }
        .weather-now {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }
        .weather-visual-badge {
            width: 64px;
            height: 64px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(203, 213, 225, 0.75);
            box-shadow: inset 0 1px 8px rgba(255,255,255,0.55);
        }
        .weather-visual-badge .weather-hero-img {
            width: 62px;
            height: 62px;
            filter: drop-shadow(0 8px 10px rgba(15,23,42,0.16));
        }
        .weather-visual-badge .weather-hero-fallback {
            width: 46px;
            height: 46px;
            background: #0f172a;
            color: #ffffff;
            font-size: 11px;
        }
        .weather-desc {
            color: #0f172a;
            font-size: 24px;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
            text-transform: capitalize;
        }
        .weather-location {
            color: #ef4444;
            font-size: 12px;
            font-weight: 800;
            margin-top: 5px;
        }
        .weather-tower {
            display: block;
            color: #334155;
            font-size: 12px;
            font-weight: 700;
            margin-top: 3px;
        }
        .weather-temp-block {
            text-align: right;
            min-width: 120px;
        }
        .weather-temp {
            color: #0f172a;
            font-size: 34px;
            line-height: 1;
            font-weight: 850;
            margin: 0;
        }
        .weather-range {
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
            margin-top: 5px;
        }
        .weather-attribute-list {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 8px;
            margin-top: 12px;
        }
        .weather-attribute-list div {
            display: block;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.70);
            padding: 8px 9px;
            color: #64748b;
            font-size: 11px;
            font-weight: 650;
        }
        .weather-attribute-list strong {
            display: block;
            margin-top: 2px;
            color: #0f172a;
            font-size: 13px;
            white-space: nowrap;
        }
        .weather-forecast-row {
            display: grid;
            grid-template-columns: minmax(0, 0.75fr) minmax(320px, 1.25fr);
            gap: 14px;
            align-items: center;
            margin-top: 12px;
        }
        .weather-forecast-copy {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.64);
            padding: 10px;
            min-width: 0;
        }
        .weather-forecast-copy span,
        .weather-forecast-copy small {
            display: block;
            color: #64748b;
            font-size: 11px;
        }
        .weather-forecast-copy strong {
            display: block;
            color: #0f172a;
            font-size: 13px;
            line-height: 1.3;
            margin: 4px 0 5px 0;
        }
        .weather-trend-svg {
            height: 88px;
            width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.52);
        }
        .weather-trend-grid line {
            stroke: rgba(100, 116, 139, 0.22);
            stroke-dasharray: 4 5;
        }
        .weather-trend-rain rect {
            fill: rgba(59, 130, 246, 0.46);
        }
        .weather-trend-line {
            fill: none;
            stroke: #f59e0b;
            stroke-width: 4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .weather-trend-points circle {
            fill: #ffffff;
            stroke: #f59e0b;
            stroke-width: 2;
        }
        .weather-forecast-strip {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding: 10px 0 4px 0;
            margin-top: 2px;
            scrollbar-width: thin;
        }
        .weather-forecast-item {
            flex: 0 0 82px;
            min-width: 82px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.78);
            background: rgba(255, 255, 255, 0.70);
            color: #0f172a;
            padding: 7px 6px;
            display: grid;
            grid-template-rows: 16px 30px 16px 15px 14px 22px 12px 22px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .weather-forecast-item span,
        .weather-forecast-item small {
            color: #64748b;
            font-size: 10px;
            font-weight: 700;
        }
        .weather-forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            filter: drop-shadow(0 6px 8px rgba(15,23,42,0.14));
        }
        .weather-forecast-item b {
            color: #0f172a;
            font-size: 12px;
            line-height: 1;
        }
        .weather-forecast-item em {
            max-width: 72px;
            color: #334155;
            font-size: 10px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .weather-forecast-item label {
            color: #475569;
            font-size: 9px;
            font-weight: 700;
            min-height: 10px;
        }
        .weather-forecast-item mark {
            display: grid;
            place-items: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            color: #0f172a;
            background: #ffffff;
            border: 3px solid #f59e0b;
            font-size: 10px;
            font-weight: 800;
            padding: 0;
        }
        .weather-precip-track {
            width: 18px;
            height: 22px;
            border-radius: 999px;
            background: rgba(203, 213, 225, 0.58);
            display: flex;
            align-items: flex-end;
            justify-content: center;
            overflow: hidden;
        }
        .weather-precip-track i {
            width: 100%;
            min-height: 3px;
            background: linear-gradient(180deg, #60a5fa, #2563eb);
        }
        .weather-storm {
            margin-top: 10px;
            border-radius: 8px;
            padding: 8px 10px;
            background: rgba(255, 237, 213, 0.82);
            border: 1px solid rgba(249, 115, 22, 0.62);
        }
        .weather-storm span {
            display: block;
            color: #9a3412;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .weather-storm strong {
            color: #0f172a;
            font-size: 12px;
        }
        .weather-footer {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-top: 9px;
            color: #64748b;
            font-size: 10px;
            flex-wrap: wrap;
        }
        @media (max-width: 1000px) {
            .weather-attribute-list {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .weather-forecast-row {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 640px) {
            .weather-glass {
                padding: 12px;
            }
            .weather-top {
                grid-template-columns: 1fr;
            }
            .weather-temp-block {
                text-align: left;
            }
            .weather-attribute-list {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .weather-desc {
                font-size: 21px;
            }
            .weather-forecast-item {
                flex-basis: 78px;
                min-width: 78px;
            }
        }
        </style>
        <div class="weather-card-wrap">
        """)
        + "".join(cards_html)
        + "</div>"
    )


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])
        theme = weather_theme_for_code(row.get("Weather Code"))
        icon_url = row.get("Weather Icon URL")
        if icon_url:
            visual_html = f"<img class='weather-icon-main' src='{icon_url}' alt='{row.get('Current Weather', 'Weather')}' />"
        else:
            visual_html = f"<div class='weather-icon-fallback'>{weather_icon_for_code(row.get('Weather Code'))}</div>"

        thunder_text = row.get("Last Thunderstorm Indication", "-")
        storm_class = "weather-storm-muted"
        if row.get("Last Thunderstorm Time"):
            thunder_text = f"{row.get('Last Thunderstorm Time')} | {row.get('Last Thunderstorm Weather', '-')}"
            storm_class = "weather-storm-active"

        forecast_items_html = ""
        temp_values = []
        for item in forecast_items[:8]:
            temp_value = item.get("temperature_c")
            try:
                temp_values.append(float(temp_value))
            except (TypeError, ValueError):
                pass
            item_icon = (
                f"<img src='{item.get('icon_url')}' alt='{item.get('description', 'Forecast')}' />"
                if item.get("icon_url")
                else f"<strong>{weather_icon_for_code(item.get('weather_code'))}</strong>"
            )
            pop_label = f"{item.get('pop_pct')}%" if item.get("pop_pct") is not None else "-"
            forecast_items_html += (
                "<div class='forecast-item'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{safe_display_number(temp_value, 0, '°C')}</b>"
                f"<em>{item.get('description', '-')}</em>"
                f"<small>{pop_label}</small>"
                "</div>"
            )

        temp_range = "-"
        if temp_values:
            temp_range = f"{safe_display_number(min(temp_values), 0, ' C')} - {safe_display_number(max(temp_values), 0, ' C')}"
        rain_bars = build_rain_chance_chart(forecast_items)
        forecast_pop = f"{forecast.get('max_pop') * 100.0:.0f}%" if forecast.get("max_pop") is not None else "-"
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        forecast_signal = "Ada potensi petir" if forecast.get("thunder_count", 0) else "Tidak ada sinyal petir kuat"

        cards_html.append(
            f"""
            <div class="weather-card weather-theme-{theme}">
                <section class="current-panel">
                    <div class="current-head">
                        <div>
                            <div class="weather-role">{row.get('Location', '-')}</div>
                            <div class="weather-title">{row.get('Tower', '-')}</div>
                            <div class="weather-temp">{safe_display_number(row.get('Temperature C'), 1, ' C')}</div>
                            <div class="weather-desc">{row.get('Current Weather', '-')}</div>
                        </div>
                        <div class="weather-visual">
                            {visual_html}
                            <div>{str(row.get('Current Weather') or '-').title()}</div>
                        </div>
                    </div>
                    <div class="metric-grid">
                        <div><span>Hujan saat ini</span><strong>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</strong></div>
                        <div><span>Kelembapan</span><strong>{safe_display_number(row.get('Humidity %'), 0, '%')}</strong></div>
                        <div><span>Tutupan awan</span><strong>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</strong></div>
                        <div><span>Kecepatan angin</span><strong>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</strong></div>
                    </div>
                    <div class="storm-box {storm_class}">
                        <span>Indikasi thunderstorm</span>
                        <strong>{thunder_text}</strong>
                    </div>
                    <div class="weather-meta">
                        <span>Kumulatif {safe_display_number(row.get('Cumulative km'), 3, ' km')}</span>
                        <span>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)}</span>
                    </div>
                    <div class="weather-time">Update cuaca: {row.get('Weather Time') or '-'} | {row.get('Weather Source', '-')}</div>
                </section>
                <section class="forecast-panel">
                    <div class="forecast-copy">
                        <span>Prakiraan 12 jam ke depan</span>
                        <strong>{forecast.get('summary', 'Forecast belum tersedia.')}</strong>
                    </div>
                    <div class="forecast-metrics">
                        <div><span>Peluang hujan tertinggi</span><strong>{forecast_pop}</strong></div>
                        <div><span>Perkiraan hujan total</span><strong>{forecast_precip}</strong></div>
                        <div><span>Rentang suhu</span><strong>{temp_range}</strong></div>
                        <div><span>Indikasi petir</span><strong>{forecast_signal}</strong></div>
                    </div>
                    <div class="forecast-strip">
                        {forecast_items_html}
                    </div>
                </section>
            </div>
            """
        )

    return (
        """
        <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .weather-card-wrap {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 14px;
        }
        .weather-card {
            display: grid;
            grid-template-columns: minmax(360px, 0.92fr) minmax(420px, 1.08fr);
            gap: 12px;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.46);
            background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 100%);
            color: #0f172a;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.11);
        }
        .weather-theme-clear { background: linear-gradient(135deg, #fff7ed 0%, #dbeafe 100%); }
        .weather-theme-cloud { background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 56%, #e2e8f0 100%); }
        .weather-theme-rain { background: linear-gradient(135deg, #eef6ff 0%, #c7d2fe 100%); }
        .weather-theme-storm { background: linear-gradient(135deg, #eef2ff 0%, #fed7aa 100%); }
        .weather-theme-mist { background: linear-gradient(135deg, #f8fafc 0%, #e5e7eb 100%); }
        .current-panel,
        .forecast-panel {
            min-width: 0;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.72);
            background: rgba(255, 255, 255, 0.56);
            padding: 12px;
        }
        .current-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 14px;
            align-items: start;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.28);
        }
        .weather-role {
            color: #ef4444;
            font-size: 12px;
            font-weight: 800;
        }
        .weather-title {
            color: #0f172a;
            font-size: 13px;
            font-weight: 800;
            margin-top: 3px;
        }
        .weather-temp {
            font-size: 40px;
            line-height: 1;
            font-weight: 850;
            margin-top: 18px;
        }
        .weather-desc {
            color: #334155;
            font-size: 16px;
            margin-top: 5px;
            text-transform: capitalize;
        }
        .weather-visual {
            width: 130px;
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.74);
            background: rgba(255, 255, 255, 0.70);
            display: grid;
            place-items: center;
            padding: 8px;
            text-align: center;
            color: #0f172a;
            font-size: 12px;
            font-weight: 800;
        }
        .weather-icon-main {
            width: 92px;
            height: 92px;
            object-fit: contain;
            filter: drop-shadow(0 8px 10px rgba(15,23,42,0.16));
        }
        .weather-icon-fallback {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #0f172a;
            color: #ffffff;
            font-size: 13px;
            font-weight: 800;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .metric-grid div,
        .forecast-metrics div,
        .forecast-item {
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.76);
            background: rgba(255, 255, 255, 0.70);
        }
        .metric-grid div,
        .forecast-metrics div {
            padding: 8px 9px;
        }
        .metric-grid span,
        .forecast-metrics span,
        .storm-box span,
        .forecast-copy span {
            display: block;
            color: #64748b;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .metric-grid strong,
        .forecast-metrics strong,
        .storm-box strong,
        .forecast-copy strong {
            color: #0f172a;
            font-size: 13px;
        }
        .storm-box {
            margin-top: 9px;
            border-radius: 8px;
            padding: 8px 10px;
            background: rgba(255, 237, 213, 0.82);
            border: 1px solid rgba(249, 115, 22, 0.62);
        }
        .storm-box span { color: #9a3412; }
        .weather-meta,
        .weather-time {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #64748b;
            font-size: 10px;
            margin-top: 8px;
            flex-wrap: wrap;
        }
        .forecast-copy {
            padding-bottom: 8px;
        }
        .forecast-copy strong {
            display: block;
            line-height: 1.3;
        }
        .forecast-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 2px;
        }
        .forecast-strip {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-top: 10px;
            scrollbar-width: thin;
        }
        .forecast-item {
            flex: 0 0 82px;
            min-width: 82px;
            padding: 7px 6px;
            display: grid;
            grid-template-rows: 16px 30px 16px 15px 14px;
            justify-items: center;
            gap: 1px;
            text-align: center;
        }
        .forecast-item span,
        .forecast-item small {
            color: #64748b;
            font-size: 10px;
            font-weight: 700;
        }
        .forecast-item img {
            width: 30px;
            height: 30px;
            object-fit: contain;
            filter: drop-shadow(0 6px 8px rgba(15,23,42,0.14));
        }
        .forecast-item b {
            color: #0f172a;
            font-size: 12px;
            line-height: 1;
        }
        .forecast-item em {
            max-width: 72px;
            color: #334155;
            font-size: 10px;
            font-style: normal;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .forecast-item strong {
            display: grid;
            place-items: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #0f172a;
            color: #ffffff;
            font-size: 9px;
        }
        @media (max-width: 980px) {
            .weather-card {
                grid-template-columns: 1fr;
            }
            .forecast-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .weather-card {
                padding: 10px;
            }
            .current-head {
                grid-template-columns: 1fr;
            }
            .weather-visual {
                width: 100%;
                grid-template-columns: auto 1fr;
                justify-content: start;
                text-align: left;
            }
            .weather-icon-main {
                width: 58px;
                height: 58px;
            }
            .metric-grid,
            .forecast-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        <div class="weather-card-wrap">
        """
        + "".join(cards_html)
        + "</div>"
    )


def weather_card_html(weather_rows):
    cards_html = []
    for row in weather_rows:
        forecast = row.get("Forecast Summary") or {}
        forecast_items = forecast.get("items", [])[:8]
        current_icon = f"<div class='weather-symbol weather-symbol-{weather_theme_for_code(row.get('Weather Code'))}'>{weather_symbol_for_code(row.get('Weather Code'))}</div>"
        forecast_cards = ""
        rain_bars = ""
        max_pop_seen = 1
        for item in forecast_items:
            try:
                max_pop_seen = max(max_pop_seen, int(item.get("pop_pct") or 0))
            except (TypeError, ValueError):
                pass
        for item in forecast_items:
            item_icon = f"<strong class='weather-symbol-mini weather-symbol-{weather_theme_for_code(item.get('weather_code'))}'>{weather_symbol_for_code(item.get('weather_code'))}</strong>"
            temp_value = item.get("temperature_c")
            try:
                pop_num = int(item.get("pop_pct") or 0)
            except (TypeError, ValueError):
                pop_num = 0
            forecast_cards += (
                "<div class='daily-card'>"
                f"<span>{item.get('time', '-')}</span>"
                f"{item_icon}"
                f"<b>{safe_display_number(temp_value, 0, '°C')}</b>"
                f"<small>{item.get('description', '-')}</small>"
                "</div>"
            )
        rain_bars = build_rain_chance_chart(forecast_items)
        forecast_pop = f"{forecast.get('max_pop') * 100.0:.0f}%" if forecast.get("max_pop") is not None else "-"
        forecast_precip = safe_display_number(forecast.get("total_precip_mm", 0.0), 2, " mm")
        trend_svg = build_weather_trend_svg(forecast_items)
        cards_html.append(
            textwrap.dedent(f"""
            <div class="dashboard-weather-card">
                <aside class="dashboard-current">
                    <div class="dashboard-location">
                        <strong>{row.get('Location', '-')}</strong>
                        <span>{row.get('Tower', '-')}</span>
                    </div>
                    <div class="dashboard-update">{row.get('Weather Time') or '-'}</div>
                    <div class="dashboard-icon">{current_icon}</div>
                    <div class="dashboard-condition">{row.get('Current Weather', '-')}</div>
                    <div class="dashboard-temp">{safe_display_number(row.get('Temperature C'), 1, '°C')}</div>
                    <div class="dashboard-feels">Terasa seperti {safe_display_number(row.get('Feels Like C'), 1, '°C')}</div>
                    <div class="dashboard-metrics-list">
                        <div><span>Hujan</span><b>{safe_display_number(row.get('Rain mm'), 2, ' mm')}</b></div>
                        <div><span>Angin</span><b>{safe_display_number(row.get('Wind km/h'), 1, ' km/h')}</b></div>
                        <div><span>Kelembapan</span><b>{safe_display_number(row.get('Humidity %'), 0, '%')}</b></div>
                        <div><span>Tutupan awan</span><b>{safe_display_number(row.get('Cloud Cover %'), 0, '%')}</b></div>
                        <div><span>Kumulatif fault</span><b>{safe_display_number(row.get('Cumulative km'), 3, ' km')}</b></div>
                    </div>
                </aside>
                <main class="dashboard-forecast">
                    <section class="dashboard-tile temp-tile">
                        <header>Tren suhu, °C</header>
                        {trend_svg}
                    </section>
                    <section class="dashboard-tile rain-tile">
                        <header>Peluang hujan, %</header>
                        <div class="rain-bars">{rain_bars}</div>
                    </section>
                    <section class="dashboard-tile summary-tile compact-summary">
                        <header>Ringkasan titik gangguan</header>
                        <strong>{forecast.get('summary', 'Forecast belum tersedia.')}</strong>
                        <div><span>Peluang hujan tertinggi</span><b>{forecast_pop}</b></div>
                        <div><span>Perkiraan hujan total</span><b>{forecast_precip}</b></div>
                    </section>
                    <section class="dashboard-daily">{forecast_cards}</section>
                    <section class="dashboard-note">
                        <small>{safe_display_number(row.get('Latitude'), 6)}, {safe_display_number(row.get('Longitude'), 6)} | {row.get('Weather Source', '-')}</small>
                    </section>
                </main>
            </div>
            """).strip()
        )
    return (
        textwrap.dedent("""
        <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; }
        .weather-card-wrap { display: grid; gap: 16px; }
        .dashboard-weather-card {
            width: min(100%, 1440px);
            margin: 0 auto;
            display: grid;
            grid-template-columns: 250px minmax(0, 1fr);
            border-radius: 14px;
            overflow: hidden;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.34);
            box-shadow: none;
        }
        .dashboard-current {
            background: linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
            padding: 18px 22px;
            border-right: 1px solid rgba(226, 232, 240, 0.9);
        }
        .dashboard-location { display: grid; gap: 4px; font-size: 12px; color: #111827; }
        .dashboard-location span, .dashboard-update, .dashboard-feels { color: #6b7280; font-size: 12px; }
        .dashboard-update { margin-top: 14px; text-align: center; }
        .dashboard-icon { width: 104px; height: 104px; display: grid; place-items: center; margin: 12px auto 4px auto; }
        .weather-symbol { font-size: 78px; line-height: 1; filter: drop-shadow(0 6px 8px rgba(15,23,42,0.18)); }
        .weather-symbol-mini { display: block; font-size: 30px; line-height: 1; background: transparent; color: inherit; }
        .weather-symbol-clear { color: #f59e0b; }
        .weather-symbol-cloud { color: #64748b; }
        .weather-symbol-rain { color: #2563eb; }
        .weather-symbol-storm { color: #d97706; }
        .weather-symbol-mist, .weather-symbol-neutral { color: #475569; }
        .dashboard-condition { text-align: center; text-transform: capitalize; font-size: 13px; color: #111827; }
        .dashboard-temp { text-align: center; font-size: 54px; line-height: 1; font-weight: 300; color: #f59e0b; margin-top: 8px; }
        .dashboard-feels { text-align: center; margin-top: 6px; }
        .dashboard-metrics-list { display: grid; gap: 9px; margin-top: 18px; }
        .dashboard-metrics-list div { display: flex; justify-content: space-between; gap: 16px; align-items: center; font-size: 13px; }
        .dashboard-metrics-list b { color: #111827; font-weight: 700; white-space: nowrap; }
        .dashboard-metrics-list b::before { display: none; }
        .dashboard-forecast { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); grid-auto-rows: min-content; gap: 12px; padding: 18px 22px; background: #ffffff; }
        .dashboard-tile, .dashboard-note { border-radius: 8px; background: #f8fafc; border: 1px solid rgba(241, 245, 249, 0.95); padding: 14px; }
        .dashboard-tile header { font-size: 14px; color: #111827; margin-bottom: 8px; }
        .weather-trend-chart { position: relative; height: 94px; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 6px; align-items: stretch; }
        .trend-slot { position: relative; display: grid; grid-template-rows: 20px 1fr 16px; align-items: end; justify-items: center; min-width: 0; color: #64748b; }
        .trend-value { color: #0f172a; font-size: 16px; font-weight: 700; line-height: 1; }
        .trend-stage { position: relative; width: 100%; height: 52px; border-bottom: 1px solid rgba(148,163,184,0.28); overflow: visible; }
        .trend-stage::before { content: ""; position: absolute; left: 50%; top: 2px; bottom: 0; border-left: 1px dashed rgba(148,163,184,0.38); }
        .trend-stage i { position: absolute; left: calc(50% - 1px); bottom: 8px; width: 2px; height: calc(8px + (var(--level) * 30px)); border-radius: 999px; background: linear-gradient(180deg, #f59e0b, #facc15); z-index: 2; }
        .trend-stage b { position: absolute; left: calc(50% - 5px); bottom: calc(6px + (var(--level) * 30px)); width: 10px; height: 10px; border-radius: 999px; background: #ffffff; border: 2px solid #f59e0b; z-index: 2; }
        .trend-slot small { font-size: 11px; color: #64748b; white-space: nowrap; }
        .weather-trend-svg { width: 100%; height: 94px; display: block; }
        .weather-trend-grid line { stroke: rgba(148, 163, 184, 0.35); stroke-dasharray: 4 5; }
        .weather-trend-rain rect { fill: rgba(56, 189, 248, 0.45); }
        .weather-trend-line { fill: none; stroke: #eab308; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
        .weather-trend-points circle { fill: #f8fafc; stroke: #f59e0b; stroke-width: 2; }
        .weather-trend-labels text { fill: #334155; font-size: 20px; font-weight: 700; }
        .weather-trend-axis text { fill: #64748b; font-size: 16px; font-weight: 600; }
        .rain-bars { position: relative; height: 94px; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); align-items: end; gap: 8px; padding-top: 8px; }
        .rain-bar-wrap { position: relative; z-index: 2; min-width: 24px; display: grid; grid-template-rows: 1fr auto auto; justify-items: center; align-items: end; gap: 4px; color: #6b7280; font-size: 10px; }
        .rain-bar-wrap i { width: 100%; max-width: 28px; height: calc(12px + (var(--rain-level) * 46px)); display: block; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #0ea5e9, #bae6fd); }
        .summary-tile { grid-column: span 2; }
        .compact-summary { display: grid; grid-template-columns: minmax(260px, 1.7fr) repeat(2, minmax(150px, 1fr)); gap: 10px; align-items: center; padding: 10px 12px; }
        .compact-summary header { margin: 0; grid-column: 1 / -1; }
        .compact-summary strong { display: block; font-size: 14px; line-height: 1.25; }
        .compact-summary div { border-radius: 8px; background: #ffffff; border: 1px solid rgba(226,232,240,0.92); padding: 8px 10px; }
        .compact-summary span { display: block; color: #6b7280; font-size: 11px; margin-bottom: 3px; }
        .dashboard-daily { grid-column: span 2; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; padding: 4px 0 0 0; }
        .daily-card { min-height: 112px; display: grid; grid-template-columns: 1fr; grid-template-rows: auto 38px auto auto; justify-items: center; align-items: center; border-radius: 10px; background: #f8fafc; border: 1px solid rgba(226,232,240,0.95); padding: 10px 8px; text-align: center; }
        .daily-card span { color: #6b7280; font-size: 11px; }
        .daily-card img, .daily-card strong { width: 38px; height: 38px; object-fit: contain; filter: saturate(1.2) contrast(1.1); }
        .daily-card b { font-size: 20px; font-weight: 500; }
        .daily-card small { color: #9ca3af; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; font-size: 10px; }
        .dashboard-note { grid-column: span 2; display: grid; gap: 4px; color: #6b7280; font-size: 11px; padding: 8px 10px; }
        .dashboard-note b { color: #111827; }
        @media (max-width: 900px) {
            .dashboard-weather-card { grid-template-columns: 1fr; }
            .dashboard-current { border-right: 0; border-bottom: 1px solid rgba(226,232,240,0.9); }
            .dashboard-forecast { grid-template-columns: 1fr; }
            .summary-tile, .dashboard-daily, .dashboard-note { grid-column: span 1; }
            .compact-summary { grid-template-columns: 1fr; }
            .compact-summary header { grid-column: auto; }
            .dashboard-daily { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        }
        @media (max-width: 520px) {
            .dashboard-current, .dashboard-forecast { padding: 20px; }
            .dashboard-temp { font-size: 46px; }
            .dashboard-daily { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        <div class="weather-card-wrap">
        """).strip()
        + "".join(cards_html)
        + "</div>"
    )
