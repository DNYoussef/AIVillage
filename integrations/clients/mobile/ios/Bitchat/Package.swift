// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "Bitchat",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "Bitchat",
            targets: ["Bitchat"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "Bitchat",
            dependencies: []),
        .testTarget(
            name: "BitchatTests",
            dependencies: ["Bitchat"]),
    ]
)
