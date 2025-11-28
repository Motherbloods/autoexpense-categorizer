const axios = require("axios");
async function testSend() {
  try {
    const webhookUrl =
      "https://n8n-ku.duckdns.org/webhook-test/5935c980-5cec-4f30-b94b-840ca3cacb1e";

    const now = new Date().toISOString();

    const data = [
      { text: "habib beli nasi goreng 15k", createdAt: now },
      { text: "beli bakso 10rb", createdAt: now },
      {
        text: "habib beli nasi padang 20k, teh manis 5k, parkir 2k",
        createdAt: now,
      },
      { text: "tadi riszal beli cilok 2k", createdAt: now },
      { text: "beli kopi dua ribu", createdAt: now },
      { text: "habib beli tiket 150 ribu", createdAt: now },
      { text: "riszal beli mie ayam 12k dan es jeruk 5k", createdAt: now },
      { text: "barusan habib beli bensin 30k", createdAt: now },
      { text: "riszal nasi goreng 12rb", createdAt: now },
      { text: "habib beli bakso 10k, riszal beli kopi 5k", createdAt: now },
      // 🔽 Tambahan variasi biar makin kuat
      { text: "habib jajan cilok 3k", createdAt: now },
      { text: "riszal makan di warung 25rb", createdAt: now },
      { text: "beli pulsa 10 ribu", createdAt: now },
      { text: "aku beli aqua 5k", createdAt: now },
      { text: "habib parkir 2000", createdAt: now },
      { text: "habib dan riszal beli es teh masing-masing 5k", createdAt: now },
      {
        text: "riszal beli bensin 30rb terus habib beli mie ayam 15rb",
        createdAt: now,
      },
      { text: "beli nasi goreng sama es teh 18rb", createdAt: now },
      { text: "habib beli snack 3k, kopi 7k", createdAt: now },
      {
        text: "riszal beli tiket bioskop seratus lima puluh ribu",
        createdAt: now,
      },
    ];

    const response = await axios.post(webhookUrl, data);
    console.log("✅ Data berhasil dikirim:", response.data);
  } catch (error) {
    console.error("❌ Gagal kirim ke n8n:", error.message);
  }
}

testSend();
