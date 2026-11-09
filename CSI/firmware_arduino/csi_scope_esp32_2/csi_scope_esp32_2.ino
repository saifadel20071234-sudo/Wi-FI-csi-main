/*
 * WiFi CSI Presence Sensing - ESP32 #2 Firmware (Vertical Link / Y-axis)
 *
 * Dedicated to Link 2: Connects to Hotspot [S24 Ultra saif_EXT]
 * Serial Baud: 921600
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "rom/ets_sys.h"

// -----------------------------------------------------------------------------
// WiFi Credentials for Access Point #2 (S24 Ultra saif_EXT)
// -----------------------------------------------------------------------------
const char* PRIMARY_SSID = "S24 Ultra saif_EXT";
const char* PRIMARY_PASS = "123456789ss";

// 25 Hz: optimal motion capture
#define CONFIG_SEND_FREQUENCY 25

static uint32_t s_count = 0;
static esp_ping_handle_t ping_handle = NULL;
static bool csi_initialized = false;
static uint8_t s_bssid[6] = {0};

// -----------------------------------------------------------------------------
// Fast CSI Receive Callback: outputs via ROM ets_printf
// -----------------------------------------------------------------------------
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) {
        return;
    }

    if (ctx && memcmp(info->mac, ctx, 6) != 0) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;

    ets_printf("CSI_DATA,%u,%02x:%02x:%02x:%02x:%02x:%02x,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,\"[",
               s_count++,
               info->mac[0], info->mac[1], info->mac[2],
               info->mac[3], info->mac[4], info->mac[5],
               rx->rssi, rx->rate, rx->sig_mode, rx->mcs, rx->cwb, rx->smoothing,
               rx->not_sounding, rx->aggregation, rx->stbc, rx->fec_coding, rx->sgi,
               rx->noise_floor, rx->ampdu_cnt, rx->channel, rx->secondary_channel,
               rx->timestamp, rx->ant, rx->sig_len, rx->sig_mode, info->len, info->first_word_invalid);

    for (int i = 0; i < info->len; i++) {
        ets_printf("%d%s", (int8_t)info->buf[i], (i == info->len - 1) ? "" : " ");
    }
    ets_printf("]\"\n");
}

// -----------------------------------------------------------------------------
// Ping Generator to sustain active CSI transmission
// -----------------------------------------------------------------------------
static void ping_success_cb(esp_ping_handle_t hdl, void *args) {}
static void ping_timeout_cb(esp_ping_handle_t hdl, void *args) {}

static void stop_ping() {
    if (ping_handle != NULL) {
        esp_ping_stop(ping_handle);
        esp_ping_delete_session(ping_handle);
        ping_handle = NULL;
    }
}

static void start_ping_gateway() {
    stop_ping();

    IPAddress gw = WiFi.gatewayIP();
    if (gw == IPAddress(0, 0, 0, 0)) {
        Serial.println("[PING] Gateway IP is 0.0.0.0, delaying ping start...");
        return;
    }

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count = 0; // Infinite
    ping_config.interval_ms = 1000 / CONFIG_SEND_FREQUENCY;
    ping_config.task_stack_size = 4096;

    ip_addr_t target_addr;
    memset(&target_addr, 0, sizeof(target_addr));
    target_addr.type = IPADDR_TYPE_V4;
    target_addr.u_addr.ip4.addr = (uint32_t)gw;

    ping_config.target_addr = target_addr;

    esp_ping_callbacks_t cbs;
    memset(&cbs, 0, sizeof(cbs));
    cbs.on_ping_success = ping_success_cb;
    cbs.on_ping_timeout = ping_timeout_cb;

    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);
    Serial.printf(">> Pinging Hotspot Gateway [%s] at %d Hz for continuous CSI\n",
                  gw.toString().c_str(), CONFIG_SEND_FREQUENCY);
}

// -----------------------------------------------------------------------------
// Activate CSI Hardware Engine
// -----------------------------------------------------------------------------
static void setup_csi() {
    if (csi_initialized) return;

    wifi_csi_config_t csi_config;
    memset(&csi_config, 0, sizeof(csi_config));
    csi_config.lltf_en = true;
    csi_config.htltf_en = false;
    csi_config.stbc_htltf2_en = false;
    csi_config.ltf_merge_en = true;
    csi_config.channel_filter_en = true;
    csi_config.manu_scale = true;
    csi_config.shift = true;

    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        memcpy(s_bssid, ap_info.bssid, 6);
        esp_wifi_set_csi_config(&csi_config);
        esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, s_bssid);
        esp_wifi_set_csi(true);
        csi_initialized = true;
        Serial.println(">> CSI Hardware Engine Activated on Link 2 (zoz)!");
    } else {
        esp_wifi_set_csi_config(&csi_config);
        esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL);
        esp_wifi_set_csi(true);
        csi_initialized = true;
        Serial.println(">> CSI Hardware Engine Activated (Unfiltered)!");
    }
}

// Scan helper
static void scan_and_report() {
    Serial.println(">> Scanning 2.4GHz WiFi networks nearby...");
    int n = WiFi.scanNetworks(false, false, false, 300);
    if (n == 0) {
        Serial.println("   No networks found at all! Is Wi-Fi radio working?");
    } else {
        Serial.printf("   Found %d network(s):\n", n);
        bool found_zoz = false;
        for (int i = 0; i < n; i++) {
            String s = WiFi.SSID(i);
            Serial.printf("    - [%s] (Ch: %d, RSSI: %d dBm)\n", s.c_str(), WiFi.channel(i), WiFi.RSSI(i));
            if (s.equalsIgnoreCase(PRIMARY_SSID)) {
                found_zoz = true;
            }
        }
        if (!found_zoz) {
            Serial.println("   [!] ALERT: Network [S24 Ultra saif_EXT] NOT found in 2.4GHz scan!");
            Serial.println("   -> Your phone hotspot is likely set to 5GHz! Change it to 2.4GHz.");
        }
    }
}

// -----------------------------------------------------------------------------
// WiFi Event Handler
// -----------------------------------------------------------------------------
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
    switch (event) {
        case ARDUINO_EVENT_WIFI_STA_CONNECTED:
            Serial.printf("[WiFi] CONNECTED to [%s]! Waiting for IP...\n", PRIMARY_SSID);
            break;

        case ARDUINO_EVENT_WIFI_STA_GOT_IP:
            Serial.printf("[WiFi] IP acquired: %s | Gateway: %s\n",
                          WiFi.localIP().toString().c_str(),
                          WiFi.gatewayIP().toString().c_str());
            setup_csi();
            start_ping_gateway();
            break;

        case ARDUINO_EVENT_WIFI_STA_DISCONNECTED: {
            uint8_t reason = info.wifi_sta_disconnected.reason;
            Serial.printf("[WiFi] Disconnected from [%s]! (Reason: %d)\n", PRIMARY_SSID, reason);
            csi_initialized = false;
            stop_ping();

            if (reason == 201) {
                Serial.println("  -> Reason 201: [S24 Ultra saif_EXT] SSID not found in 2.4GHz band!");
                scan_and_report();
            } else if (reason == 36 || reason == 15 || reason == 202) {
                Serial.println("  -> Reason 36/15: Auth handshake failed (WPA3 / PMF issue).");
                Serial.println("  -> Change Hotspot Security from WPA3 to WPA2-Personal on the phone!");
            } else if (reason == 2) {
                Serial.println("  -> Reason 2: Auth expired! Reconnecting...");
            }
            break;
        }

        default:
            break;
    }
}

void setup() {
    Serial.begin(921600);
    delay(1000);
    Serial.println("\n==========================================");
    Serial.println("   WiFi CSI Presence - ESP32 #2 (Link 2)");
    Serial.println("   Connecting exclusively to: [S24 Ultra saif_EXT]");
    Serial.println("==========================================");

    WiFi.onEvent(onWiFiEvent);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    WiFi.setAutoReconnect(true);

    // Configure PMF (Protected Management Frames) for modern Android compatibility
    wifi_config_t conf;
    memset(&conf, 0, sizeof(conf));
    strncpy((char*)conf.sta.ssid, PRIMARY_SSID, sizeof(conf.sta.ssid));
    strncpy((char*)conf.sta.password, PRIMARY_PASS, sizeof(conf.sta.password));
    conf.sta.pmf_cfg.capable = true;
    conf.sta.pmf_cfg.required = false;
    esp_wifi_set_config(WIFI_IF_STA, &conf);

    Serial.printf("Connecting to Hotspot [%s] ...\n", PRIMARY_SSID);
    WiFi.begin(PRIMARY_SSID, PRIMARY_PASS);
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long last_retry = 0;
        if (millis() - last_retry > 6000) {
            last_retry = millis();
            Serial.printf("Retrying connection to [%s] ...\n", PRIMARY_SSID);
            WiFi.disconnect();
            delay(200);
            WiFi.begin(PRIMARY_SSID, PRIMARY_PASS);
        }
    }
    delay(1000);
}
