/*
 * WiFi CSI Presence Sensing - ESP32 #2 Firmware (Vertical Link / Y-axis)
 *
 * Identical to ESP32 #1 firmware but connects to AP2 (second router/access point).
 * This ESP forms the VERTICAL sensing link. Place it perpendicular to ESP32 #1.
 *
 * Hardware placement:
 *   ESP32 #1  ←→  AP1 : Horizontal beam (measures X position)
 *   ESP32 #2  ←→  AP2 : Vertical beam   (measures Y position)
 *
 * Serial Baud: 921600
 * Flash this to your SECOND ESP32 on whatever COM port it appears as.
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "rom/ets_sys.h"

// -----------------------------------------------------------------------------
// WiFi Credentials for AP2 (Second Router / Access Point)
// Change these to match your second router's SSID and password.
// -----------------------------------------------------------------------------
const char* PRIMARY_SSID = "CSI_AP2";
const char* PRIMARY_PASS = "csicsi2024";

// Fallback: if AP2 not available, try the mobile hotspot as backup
const char* BACKUP_SSID  = "S24 Ultra saif";
const char* BACKUP_PASS  = "123456789ss";

// 25 Hz: fast enough to catch human movement (walking ~1-2 m/s)
#define CONFIG_SEND_FREQUENCY 25

static uint32_t s_count = 0;
static esp_ping_handle_t ping_handle = NULL;
static bool csi_initialized = false;
static uint8_t s_bssid[6] = {0};
static bool using_backup = false;

// -----------------------------------------------------------------------------
// Fast CSI Receive Callback: outputs via ROM ets_printf (non-blocking, fast)
// Output format matches Python parse_csi_line in csi_reader.py
// -----------------------------------------------------------------------------
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) {
        return;
    }

    // BSSID filter: only process packets from the connected AP
    if (ctx && memcmp(info->mac, ctx, 6) != 0) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;

    // Identical format to ESP32 #1 — Python parser handles both transparently
    ets_printf("CSI_DATA,%u,%02x:%02x:%02x:%02x:%02x:%02x,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,\"[",
               s_count++,
               info->mac[0], info->mac[1], info->mac[2],
               info->mac[3], info->mac[4], info->mac[5],
               rx->rssi, rx->rate, rx->sig_mode, rx->mcs, rx->cwb, rx->smoothing,
               rx->not_sounding, rx->aggregation, rx->stbc, rx->fec_coding, rx->sgi,
               rx->noise_floor, rx->ampdu_cnt, rx->channel, rx->secondary_channel,
               rx->timestamp, rx->ant, rx->sig_len, rx->sig_mode, info->len, info->first_word_invalid);

    for (int i = 0; i < info->len; i++) {
        ets_printf("%d%s", info->buf[i], (i < info->len - 1) ? "," : "");
    }
    ets_printf("]\"\n");
}

// -----------------------------------------------------------------------------
// Ping Generator: Sends continuous ICMP Echo to gateway to generate CSI traffic
// -----------------------------------------------------------------------------
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
    ping_config.data_size = 1;

    ping_config.target_addr.u_addr.ip4.addr = static_cast<uint32_t>(gw);
    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

    esp_ping_callbacks_t cbs = { 0 };
    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);
    Serial.printf(">> Ping started to %s at %d Hz\n", gw.toString().c_str(), CONFIG_SEND_FREQUENCY);
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
        Serial.println(">> CSI Hardware Engine Activated (Filtered to AP2 BSSID)!");
    } else {
        // Fallback: unfiltered
        esp_wifi_set_csi_config(&csi_config);
        esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL);
        esp_wifi_set_csi(true);
        csi_initialized = true;
        Serial.println(">> CSI Hardware Engine Activated (Unfiltered)!");
    }
}

// -----------------------------------------------------------------------------
// WiFi Event Handler
// -----------------------------------------------------------------------------
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
    switch (event) {
        case ARDUINO_EVENT_WIFI_STA_CONNECTED:
            Serial.printf("[WiFi] Connected to %s!\n", using_backup ? BACKUP_SSID : PRIMARY_SSID);
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
            Serial.printf("[WiFi] Disconnected from %s! (Reason: %d)\n",
                          using_backup ? BACKUP_SSID : PRIMARY_SSID, reason);
            if (reason == 201) {
                Serial.println("  -> Reason 201: SSID not found! Is AP2 ON and broadcasting 2.4GHz?");
                // Try backup if primary fails
                if (!using_backup) {
                    Serial.printf("  -> Trying backup: [%s] ...\n", BACKUP_SSID);
                    using_backup = true;
                    csi_initialized = false;
                    stop_ping();
                    WiFi.disconnect();
                    delay(500);
                    WiFi.begin(BACKUP_SSID, BACKUP_PASS);
                    return;
                }
            } else if (reason == 15 || reason == 202) {
                Serial.println("  -> Reason 15/202: 4-Way Handshake Timeout! Check AP2 password or PMF settings.");
            } else if (reason == 2) {
                Serial.println("  -> Reason 2: Auth Expired! Reconnecting...");
            }
            csi_initialized = false;
            stop_ping();
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
    Serial.println("   WiFi CSI Presence - ESP32 #2 (V-Link)");
    Serial.println("   Vertical Beam: Measures Y-axis position");
    Serial.println("==========================================");

    WiFi.onEvent(onWiFiEvent);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    WiFi.setAutoReconnect(true);

    Serial.printf("Connecting to AP2 [%s] ...\n", PRIMARY_SSID);
    WiFi.begin(PRIMARY_SSID, PRIMARY_PASS);
}

void loop() {
    // Keep WiFi alive: retry connection if dropped
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long last_retry = 0;
        if (millis() - last_retry > 5000) {
            last_retry = millis();
            const char* ssid = using_backup ? BACKUP_SSID : PRIMARY_SSID;
            const char* pass = using_backup ? BACKUP_PASS  : PRIMARY_PASS;
            Serial.printf("Retrying connection to [%s] ...\n", ssid);
            WiFi.disconnect();
            WiFi.begin(ssid, pass);
        }
    }
    delay(1000);
}
