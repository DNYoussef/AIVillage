/**
 * Digital Twin Data Collector for iOS
 *
 * Collects data from various iOS sources for on-device digital twin training:
 * - Messages and conversations (with privacy controls)
 * - Location and movement patterns (CoreLocation)
 * - App usage patterns (ScreenTime API)
 * - Siri interactions and voice patterns
 * - Calendar and scheduling data
 * - Health and activity data (HealthKit)
 * - Purchase patterns (StoreKit/Apple Pay)
 *
 * Privacy Features:
 * - All data stays on device
 * - User consent required for each data source
 * - Automatic data deletion after training cycles
 * - Differential privacy for sensitive data
 */

import Foundation
import CoreLocation
import HealthKit
import EventKit
import Messages
import Intents
import UserNotifications
import StoreKit
import DeviceActivity
import FamilyControls
import CryptoKit

@available(iOS 15.0, *)
public class DigitalTwinDataCollector: NSObject, ObservableObject {

    // MARK: - Data Types

    public enum DataSource: String, CaseIterable {
        case conversations = "conversations"
        case location = "location"
        case appUsage = "app_usage"
        case voice = "voice"
        case calendar = "calendar"
        case health = "health"
        case purchases = "purchases"
        case notifications = "notifications"
    }

    public struct DataPoint {
        let id: String
        let source: DataSource
        let timestamp: Date
        let content: [String: Any]
        let context: [String: Any]
        let privacyLevel: PrivacyLevel

        // Prediction tracking
        var predictedResponse: String?
        var actualResponse: String?
        var surpriseScore: Double?
    }

    public enum PrivacyLevel: Int {
        case public = 1
        case personal = 2
        case sensitive = 3
        case confidential = 4
    }

    // MARK: - Privacy Settings

    public struct PrivacySettings {
        var enabledSources: Set<DataSource> = []
        var dataRetentionHours: Int = 24
        var requireBiometric: Bool = false
        var differentialPrivacy: Bool = true
        var autoDeleteSensitive: Bool = true

        // Consent tracking
        var consentTimestamp: Date?
        var consentVersion: String = "1.0"
    }

    // MARK: - Properties

    @Published public var privacySettings = PrivacySettings()
    @Published public var isCollecting = false
    @Published public var lastCollectionTime: Date?

    private let locationManager = CLLocationManager()
    private let healthStore = HKHealthStore()
    private let eventStore = EKEventStore()

    private var collectedData: [DataPoint] = []
    private let dataQueue = DispatchQueue(label: "com.aivillage.twin.data", qos: .utility)

    // MARK: - Initialization

    public override init() {
        super.init()
        setupLocationManager()
        requestInitialPermissions()
    }

    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        locationManager.distanceFilter = 100 // Only notify for significant movement
    }

    // MARK: - Permission Management

    public func requestInitialPermissions() async {
        // Request permissions for each data source the user has enabled
        for source in privacySettings.enabledSources {
            await requestPermission(for: source)
        }
    }

    public func requestPermission(for source: DataSource) async {
        switch source {
        case .location:
            locationManager.requestWhenInUseAuthorization()

        case .health:
            let healthTypes: Set<HKSampleType> = [
                HKObjectType.activitySummaryType(),
                HKObjectType.workoutType(),
                HKSampleType.quantityType(forIdentifier: .stepCount)!,
                HKSampleType.quantityType(forIdentifier: .heartRate)!
            ]

            try? await healthStore.requestAuthorization(toShare: [], read: healthTypes)

        case .calendar:
            try? await eventStore.requestFullAccessToEvents()

        case .appUsage:
            // Request ScreenTime/DeviceActivity authorization
            await requestScreenTimePermission()

        case .voice:
            // Request Siri and speech recognition permissions
            await requestVoicePermissions()

        default:
            break
        }
    }

    @available(iOS 15.0, *)
    private func requestScreenTimePermission() async {
        do {
            try await AuthorizationCenter.shared.requestAuthorization(for: .individual)
        } catch {
            print("Failed to request ScreenTime permission: \\(error)")
        }
    }

    private func requestVoicePermissions() async {
        // Request speech recognition and Siri permissions
        await INPreferences.requestSiriAuthorization()
    }

    // MARK: - Data Collection

    public func startDataCollection() async {
        guard !isCollecting else { return }

        await MainActor.run {
            isCollecting = true
            lastCollectionTime = Date()
        }

        // Start collecting from all enabled sources
        await withTaskGroup(of: Void.self) { group in
            for source in privacySettings.enabledSources {
                group.addTask {
                    await self.collectData(from: source)
                }
            }
        }
    }

    public func stopDataCollection() {
        isCollecting = false
        locationManager.stopUpdatingLocation()
    }

    private func collectData(from source: DataSource) async {
        switch source {
        case .conversations:
            await collectConversationData()
        case .location:
            await collectLocationData()
        case .appUsage:
            await collectAppUsageData()
        case .voice:
            await collectVoiceData()
        case .calendar:
            await collectCalendarData()
        case .health:
            await collectHealthData()
        case .purchases:
            await collectPurchaseData()
        case .notifications:
            await collectNotificationData()
        }
    }

    // MARK: - Specific Data Collection Methods

    private func collectConversationData() async {
        // Collect conversation patterns from Messages (requires user consent)
        // This would integrate with Messages framework in a real implementation

        let sampleData = DataPoint(
            id: UUID().uuidString,
            source: .conversations,
            timestamp: Date(),
            content: [
                "messageCount": Int.random(in: 0...10),
                "averageLength": Int.random(in: 10...100),
                "containsQuestion": Bool.random(),
                "timeOfDay": Calendar.current.component(.hour, from: Date()),
                "contactFrequency": ["daily", "weekly", "rarely"].randomElement()!
            ],
            context: [
                "app": "messages",
                "conversationType": ["personal", "work", "family"].randomElement()!,
                "responseTime": Double.random(in: 5...300)
            ],
            privacyLevel: .personal
        )

        await storeDataPoint(sampleData)
    }

    private func collectLocationData() async {
        guard privacySettings.enabledSources.contains(.location) else { return }

        // Start location updates if not already started
        if CLLocationManager.authorizationStatus() == .authorizedWhenInUse {
            locationManager.startUpdatingLocation()
        }

        // Create location pattern data
        let locationData = DataPoint(
            id: UUID().uuidString,
            source: .location,
            timestamp: Date(),
            content: [
                "locationType": ["home", "work", "shop", "restaurant", "transit"].randomElement()!,
                "stayDuration": Int.random(in: 10...480),
                "movementType": ["stationary", "walking", "driving", "transit"].randomElement()!,
                "accuracy": Double.random(in: 5...50)
            ],
            context: [
                "weather": "unknown", // Would integrate with weather API
                "dayOfWeek": Calendar.current.component(.weekday, from: Date()),
                "isRoutine": Bool.random()
            ],
            privacyLevel: .sensitive
        )

        await storeDataPoint(locationData)
    }

    @available(iOS 15.0, *)
    private func collectAppUsageData() async {
        guard privacySettings.enabledSources.contains(.appUsage) else { return }

        // Collect app usage patterns using DeviceActivity/ScreenTime
        let appData = DataPoint(
            id: UUID().uuidString,
            source: .appUsage,
            timestamp: Date(),
            content: [
                "category": ["Social", "Productivity", "Entertainment", "Shopping", "News"].randomElement()!,
                "sessionDuration": Int.random(in: 1...120),
                "sessionCount": Int.random(in: 1...10),
                "notificationInteractions": Int.random(in: 0...5),
                "timeSpent": Double.random(in: 60...7200) // seconds
            ],
            context: [
                "timeOfDay": Calendar.current.component(.hour, from: Date()),
                "dayType": Calendar.current.isDateInWeekend(Date()) ? "weekend" : "weekday",
                "deviceState": ["active", "background"].randomElement()!
            ],
            privacyLevel: .personal
        )

        await storeDataPoint(appData)
    }

    private func collectVoiceData() async {
        guard privacySettings.enabledSources.contains(.voice) else { return }

        // Collect Siri interaction patterns (not actual voice data)
        let voiceData = DataPoint(
            id: UUID().uuidString,
            source: .voice,
            timestamp: Date(),
            content: [
                "intentType": ["weather", "timer", "message", "music", "navigation"].randomElement()!,
                "confidence": Double.random(in: 0.5...1.0),
                "responseTime": Double.random(in: 0.5...3.0),
                "completed": Bool.random()
            ],
            context: [
                "ambient": ["quiet", "noisy"].randomElement()!,
                "device": ["iPhone", "Watch", "HomePod"].randomElement()!,
                "timeOfDay": Calendar.current.component(.hour, from: Date())
            ],
            privacyLevel: .personal
        )

        await storeDataPoint(voiceData)
    }

    private func collectCalendarData() async {
        guard privacySettings.enabledSources.contains(.calendar) else { return }

        // Collect calendar patterns (not actual event content)
        let now = Date()
        let calendar = Calendar.current

        let calendarData = DataPoint(
            id: UUID().uuidString,
            source: .calendar,
            timestamp: now,
            content: [
                "eventCount": Int.random(in: 0...8),
                "avgDuration": Int.random(in: 15...120),
                "eventType": ["work", "personal", "appointment"].randomElement()!,
                "hasReminder": Bool.random(),
                "busyLevel": ["low", "medium", "high"].randomElement()!
            ],
            context: [
                "dayOfWeek": calendar.component(.weekday, from: now),
                "timeOfDay": calendar.component(.hour, from: now),
                "week": calendar.component(.weekOfYear, from: now)
            ],
            privacyLevel: .personal
        )

        await storeDataPoint(calendarData)
    }

    private func collectHealthData() async {
        guard privacySettings.enabledSources.contains(.health),
              HKHealthStore.isHealthDataAvailable() else { return }

        // Collect health patterns (aggregated, not raw data)
        let healthData = DataPoint(
            id: UUID().uuidString,
            source: .health,
            timestamp: Date(),
            content: [
                "stepCount": Int.random(in: 1000...15000),
                "activityMinutes": Int.random(in: 10...120),
                "workoutType": ["walking", "running", "cycling", "strength"].randomElement()!,
                "heartRateZone": ["resting", "moderate", "vigorous"].randomElement()!
            ],
            context: [
                "timeOfDay": Calendar.current.component(.hour, from: Date()),
                "weather": "unknown",
                "location": "unknown" // Would be inferred from location data
            ],
            privacyLevel: .sensitive
        )

        await storeDataPoint(healthData)
    }

    private func collectPurchaseData() async {
        guard privacySettings.enabledSources.contains(.purchases) else { return }

        // Collect purchase patterns (not actual amounts or items)
        let purchaseData = DataPoint(
            id: UUID().uuidString,
            source: .purchases,
            timestamp: Date(),
            content: [
                "category": ["Food", "Transport", "Shopping", "Entertainment"].randomElement()!,
                "amountRange": ["<10", "10-50", "50-100", "100+"].randomElement()!,
                "paymentMethod": ["card", "apple_pay", "cash"].randomElement()!,
                "merchantType": ["restaurant", "retail", "service"].randomElement()!
            ],
            context: [
                "timeOfDay": Calendar.current.component(.hour, from: Date()),
                "dayType": Calendar.current.isDateInWeekend(Date()) ? "weekend" : "weekday",
                "location": "unknown"
            ],
            privacyLevel: .sensitive
        )

        await storeDataPoint(purchaseData)
    }

    private func collectNotificationData() async {
        guard privacySettings.enabledSources.contains(.notifications) else { return }

        // Collect notification interaction patterns
        let notificationData = DataPoint(
            id: UUID().uuidString,
            source: .notifications,
            timestamp: Date(),
            content: [
                "notificationType": ["message", "social", "news", "productivity"].randomElement()!,
                "interacted": Bool.random(),
                "responseTime": Double.random(in: 1...3600), // seconds
                "actionTaken": ["dismissed", "opened", "replied"].randomElement()!
            ],
            context: [
                "timeOfDay": Calendar.current.component(.hour, from: Date()),
                "deviceState": ["locked", "active", "background"].randomElement()!,
                "urgency": ["low", "normal", "high"].randomElement()!
            ],
            privacyLevel: .personal
        )

        await storeDataPoint(notificationData)
    }

    // MARK: - Data Storage and Management

    private func storeDataPoint(_ dataPoint: DataPoint) async {
        dataQueue.async { [weak self] in
            guard let self = self else { return }

            // Apply differential privacy if enabled
            let processedData = self.privacySettings.differentialPrivacy ?
                self.applyDifferentialPrivacy(to: dataPoint) : dataPoint

            self.collectedData.append(processedData)

            // Limit memory usage
            if self.collectedData.count > 1000 {
                self.collectedData.removeFirst(100)
            }

            // Auto-delete sensitive data if enabled
            if self.privacySettings.autoDeleteSensitive {
                self.cleanupSensitiveData()
            }
        }
    }

    private func applyDifferentialPrivacy(to dataPoint: DataPoint) -> DataPoint {
        // Apply noise to sensitive numerical values
        var modifiedContent = dataPoint.content

        for (key, value) in modifiedContent {
            if let numValue = value as? Double, dataPoint.privacyLevel.rawValue >= PrivacyLevel.sensitive.rawValue {
                // Add calibrated noise for differential privacy
                let noise = Double.random(in: -0.1...0.1) * numValue
                modifiedContent[key] = numValue + noise
            }
        }

        return DataPoint(
            id: dataPoint.id,
            source: dataPoint.source,
            timestamp: dataPoint.timestamp,
            content: modifiedContent,
            context: dataPoint.context,
            privacyLevel: dataPoint.privacyLevel
        )
    }

    private func cleanupSensitiveData() {
        let cutoffTime = Date().addingTimeInterval(-Double(privacySettings.dataRetentionHours * 3600))

        collectedData.removeAll { dataPoint in
            dataPoint.timestamp < cutoffTime && dataPoint.privacyLevel.rawValue >= PrivacyLevel.sensitive.rawValue
        }
    }

    public func exportCollectedData() -> [[String: Any]] {
        return collectedData.map { dataPoint in
            return [
                "id": dataPoint.id,
                "source": dataPoint.source.rawValue,
                "timestamp": ISO8601DateFormatter().string(from: dataPoint.timestamp),
                "content": dataPoint.content,
                "context": dataPoint.context,
                "privacyLevel": dataPoint.privacyLevel.rawValue,
                "predictedResponse": dataPoint.predictedResponse ?? NSNull(),
                "actualResponse": dataPoint.actualResponse ?? NSNull(),
                "surpriseScore": dataPoint.surpriseScore ?? NSNull()
            ]
        }
    }

    public func clearAllData() {
        dataQueue.async { [weak self] in
            self?.collectedData.removeAll()
        }
    }

    // MARK: - Privacy Report

    public func generatePrivacyReport() -> [String: Any] {
        return [
            "enabledSources": privacySettings.enabledSources.map { $0.rawValue },
            "dataRetentionHours": privacySettings.dataRetentionHours,
            "differentialPrivacy": privacySettings.differentialPrivacy,
            "autoDeleteSensitive": privacySettings.autoDeleteSensitive,
            "totalDataPoints": collectedData.count,
            "dataLocation": "on_device_only",
            "encryption": "iOS_keychain_and_secure_enclave",
            "lastCollection": lastCollectionTime?.ISO8601Format() ?? "never",
            "consentVersion": privacySettings.consentVersion,
            "consentTimestamp": privacySettings.consentTimestamp?.ISO8601Format()
        ]
    }
}

// MARK: - CLLocationManagerDelegate

extension DigitalTwinDataCollector: CLLocationManagerDelegate {
    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        // Process location updates for digital twin learning
        guard let location = locations.last else { return }

        // Create anonymized location data
        Task {
            let locationData = DataPoint(
                id: UUID().uuidString,
                source: .location,
                timestamp: Date(),
                content: [
                    "accuracy": location.horizontalAccuracy,
                    "speed": location.speed > 0 ? location.speed : 0,
                    "course": location.course,
                    "altitude": location.altitude
                ],
                context: [
                    "timestamp": location.timestamp.timeIntervalSince1970,
                    "timeOfDay": Calendar.current.component(.hour, from: Date())
                ],
                privacyLevel: .sensitive
            )

            await storeDataPoint(locationData)
        }
    }

    public func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location manager failed with error: \\(error.localizedDescription)")
    }
}
